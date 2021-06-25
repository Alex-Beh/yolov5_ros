#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/opencv.hpp>

#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

//TODO: batch size from here
static float data[1 * 3 * INPUT_H * INPUT_W];
static float prob[1 * OUTPUT_SIZE];


class YOLOV5_ROS{
    public:
    YOLOV5_ROS(ros::NodeHandle &nh): nodeHandle_(nh)
    {
        image_transport::ImageTransport it(nodeHandle_);
        m_image_pub = it.advertise("/yolov5_video", 1);
        
        m_rgb_img_sub.subscribe(nodeHandle_,"/spencer/sensors/rgbd_front_top/color/image_raw", 1);
        m_depth_img_sub.subscribe(nodeHandle_, "/spencer/sensors/rgbd_front_top/aligned_depth_to_color/image_raw", 1);
        sync.reset(new Synchronizer(SyncPolicy(10), m_rgb_img_sub,m_depth_img_sub));
        sync->registerCallback(boost::bind(&YOLOV5_ROS::imageCallback, this, _1, _2));

        std::string engine_name;
        nh.param("model_name", engine_name, std::string("gesture.engine"));

        nh.param("nms_threshold", m_nms_thresh, 0.4);
        nh.param("conf_thresh", m_conf_thresh, 0.5);
        nh.param("batch_size", m_batch_size, 1);

        ROS_INFO("\033[1;32m----> engine_name: %s\033[0m",engine_name.c_str());
        ROS_INFO("\033[1;32m----> nms_threshold: %f\033[0m",m_nms_thresh);
        ROS_INFO("\033[1;32m----> conf_thresh: %f\033[0m",m_conf_thresh);
        ROS_INFO("\033[1;32m----> batch_size: %d\033[0m",m_batch_size);

        if(!readFile(engine_name))
            return;
        initEngine();
    }

    bool readFile(std::string engineFileName){
         file= std::ifstream(engineFileName, std::ios::binary);
        if (!file.good()) {
            std::cerr << "read " << engineFileName << " error!" << std::endl;
            return false;
        }
        return true;
    }

    void initEngine(){
        char *trtModelStream = nullptr;
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();

        // prepare input data ---------------------------
        printf("m_batch_size: %d --- data_size: %d!!!\n",m_batch_size,m_batch_size * 3 * INPUT_H * INPUT_W);
        runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(engine != nullptr);
        
        context = engine->createExecutionContext();
        
        assert(context != nullptr);
        delete[] trtModelStream;
        assert(engine->getNbBindings() == 2);

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        m_inputIndex = inputIndex;
        m_outputIndex = outputIndex;

        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc(&buffers[inputIndex], m_batch_size * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[outputIndex], m_batch_size * OUTPUT_SIZE * sizeof(float)));
        // Create stream

        CUDA_CHECK(cudaStreamCreate(&stream));

        // =======================================================
        //TODO: Get the information from camera_info
        //double dWidth = cap.get(CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
        //double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
        double dWidth = 640.0; //get the width of frames of the video
        double dHeight = 480.0; //get the height of frames of the video

        std::cout << "Resolution of the video : " << dWidth << " x " << dHeight << std::endl;
    }

    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& rgb_image_msg , const sensor_msgs::ImageConstPtr& depth_image_msg){
        cv_bridge::CvImagePtr rgb_cv_ptr,depth_cv_ptr;
        try
        {
            rgb_cv_ptr = cv_bridge::toCvCopy(rgb_image_msg, sensor_msgs::image_encodings::BGR8);
            depth_cv_ptr = cv_bridge::toCvCopy(depth_image_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
        }

        if (rgb_cv_ptr->image.empty() || depth_cv_ptr->image.empty()) return;
        cv::Mat pr_img = preprocess_img(rgb_cv_ptr->image, INPUT_W, INPUT_H); // letterbox BGR to RGB & resize

        int i=0;
        int fcount = 1;

        // This for loop is convert the cv::Mat into 1D Float array and pass into doInteference
        for (int row = 0; row < INPUT_H; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < INPUT_W; ++col) {
                data[i] = (float)uc_pixel[2] / 255.0;
                data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }

        // Run inference
        doInference(*context, stream, buffers, data, prob, m_batch_size);
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);

        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], m_conf_thresh, m_nms_thresh);
        }

        float pixel_distance;
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(pr_img, res[j].bbox);
                cv::rectangle(pr_img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);

                double mid_x = r.x + (r.width / 2);
                double mid_y = r.y + (r.height / 2);
                pixel_distance = 0.001 * (depth_cv_ptr->image.at<u_int16_t>(mid_y, mid_x));
                char str[200];
                sprintf(str,"%s : %.2f m",class_name[(int)res[j].class_id],pixel_distance);
                cv::circle(pr_img, cv::Point(mid_x, mid_y), 3, cv::Scalar(0, 0, 255), -1);
                cv::putText(pr_img,str, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                printf("pixel_distance of %s: %f\n",class_name[(int)res[j].class_id],pixel_distance);
            }
        }

        sensor_msgs::Image img_msg;
        std_msgs::Header header; 
        header.stamp = ros::Time::now();
        img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, pr_img);
        img_bridge.toImageMsg(img_msg);
        m_image_pub.publish(img_msg);
    }

    void clearMemory(){
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(buffers[m_inputIndex]));
        CUDA_CHECK(cudaFree(buffers[m_outputIndex]));
        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
    }

    private:
        std::ifstream file;

        IRuntime* runtime;
        ICudaEngine* engine;
        IExecutionContext* context;

        int m_inputIndex,m_outputIndex;

        cudaStream_t stream;

        void* buffers[2];

        cv_bridge::CvImage img_bridge;
        image_transport::Publisher m_image_pub;

        ros::NodeHandle nodeHandle_;
        message_filters::Subscriber<sensor_msgs::Image> m_rgb_img_sub;
        message_filters::Subscriber<sensor_msgs::Image> m_depth_img_sub;
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
        typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;
        boost::shared_ptr<Synchronizer> sync;

        const char *class_name[80] = {"palm", "fist"};
        // const char *class_name[80] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        //         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        //         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        //         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        //         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        //         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        //         "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        //         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        //         "hair drier", "toothbrush"};

        double m_nms_thresh,m_conf_thresh;
        int m_batch_size;
};


int main(int argc, char* argv[])
{
    ros::init(argc, argv, "video_inteference_yolov5");
    ros::NodeHandle nh("~");

    // YOLOV5_ROS yolov5_gesture =YOLOV5_ROS(nh);
    YOLOV5_ROS YOLOV5_ROS(nh);
    ros::spin();

    YOLOV5_ROS.clearMemory();

    return 0;
}

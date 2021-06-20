#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

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
    YOLOV5_ROS(ros::NodeHandle nh){
        image_transport::ImageTransport it(nh);
        m_image_pub = it.advertise("/yolov5_video", 1);
        m_sub = it.subscribe("/usb_cam/image_raw", 1, &YOLOV5_ROS::imageCallback,this);
        
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

    void imageCallback(const sensor_msgs::ImageConstPtr& msg){
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
        }

        if (cv_ptr->image.empty()) return;
        cv::Mat pr_img = preprocess_img(cv_ptr->image, INPUT_W, INPUT_H); // letterbox BGR to RGB & resize

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

        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(pr_img, res[j].bbox);
                cv::rectangle(pr_img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(pr_img, class_name[(int)res[j].class_id], cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
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
        image_transport::Subscriber m_sub;
        // const char *class_name[80] = {"palm", "fist"};
        const char *class_name[80] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush"};

        double m_nms_thresh,m_conf_thresh;
        int m_batch_size;
};


int main(int argc, char* argv[])
{
    ros::init(argc, argv, "video_inteference_yolov5");
    ros::NodeHandle nh("~");

    YOLOV5_ROS yolov5_gesture =YOLOV5_ROS(nh);
    ros::spin();

    yolov5_gesture.clearMemory();

    return 0;
}

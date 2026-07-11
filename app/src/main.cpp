#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "face_pipeline/face_embedder.h"
#include "face_pipeline/face_pipeline.h"
#include "face_pipeline/result_sink.h"
#include "network_camera_receiver.h"

namespace {

void drawTracks(cv::Mat& frame, const std::vector<face_pipeline::TrackedFace>& tracks) {
    for (const face_pipeline::TrackedFace& track : tracks) {
        const face_pipeline::FaceObservation& face = track.observation;
        cv::rectangle(frame, face.bbox, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "track " + std::to_string(track.track_id),
                    cv::Point(face.bbox.x, std::max(15, face.bbox.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        for (const cv::Point2f& landmark : face.landmarks) {
            cv::circle(frame, landmark, 2, cv::Scalar(0, 0, 255), -1);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = argc > 1
        ? argv[1]
        : std::string(SOURCE_ROOT) + "/models/face_recognition_sface_2021dec.onnx";

    try {
        std::unique_ptr<face_pipeline::FaceEmbedder> embedder(
            new face_pipeline::SFaceEmbedder(model_path));
        std::unique_ptr<face_pipeline::ResultSink> sink(
            new face_pipeline::LoggingResultSink(std::cout));
        face_pipeline::FacePipeline pipeline(std::move(embedder), std::move(sink));

        NetworkCameraReceiver receiver("127.0.0.1", 5000);
        if (!receiver.connect()) return 1;

        pipeline.start();
        receiver.start();

        const std::string window_name = "Face Detection and Features";
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        std::cout << "Face pipeline started; press ESC to exit." << std::endl;

        NetworkFrame network_frame;
        while (true) {
            if (receiver.getLatestFrame(network_frame) && !network_frame.image.empty()) {
                face_pipeline::FramePacket packet;
                packet.frame = network_frame.image;
                packet.frame_id = network_frame.frame_id;
                packet.captured_at = network_frame.captured_at;
                packet.captured_at_unix_ms = network_frame.captured_at_unix_ms;
                pipeline.submit(packet);

                cv::Mat display = network_frame.image.clone();
                drawTracks(display, pipeline.latestTracks());
                cv::imshow(window_name, display);
            }
            if (cv::waitKey(30) == 27) break;
        }

        receiver.stop();
        pipeline.stop();
        cv::destroyAllWindows();
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV initialization failed: " << error.what() << std::endl;
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Initialization failed: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}

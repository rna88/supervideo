#include </usr/include/opencv/cv.h>
#include </usr/include/opencv/ml.h>
#include </usr/include/opencv/cxcore.h>
#include </usr/include/opencv/highgui.h>
 
//using namespace cv;
//using namespace std;

struct channels
{
  cv::Mat Y,U,V;
};

void test(std::vector<cv::Mat>& frames)
{
  imshow("FrameFirst",frames[1]); 
  imshow("FrameSecond",frames[5]); 
  imshow("Difference",frames[1] - frames[5]);  
}

channels convertToYUV(cv::Mat& image)
{
  cv::cvtColor(image, image, CV_RGB2YCrCb);
  cv::Mat channel[3];
  cv::split(image, channel);

  channels frame;

  frame.Y = channel[0];
  frame.U = channel[1];
  frame.V = channel[2];

  return frame;
}

void interpolate(std::vector<cv::Mat>& frames)
{

  channels frame = convertToYUV(frames[0]);
  imshow("Y", frame.Y);
  imshow("Cr", frame.U);
  imshow("Cb", frame.V);

}

void readVideo(std::vector<cv::Mat>& inputFrames, char* fileName)
{
  cv::VideoCapture cap;
  cap.open(fileName);
  if (!cap.isOpened()) std::cout << "FAILED TO OPEN VIDEO.";
  
  cv::Mat frame;
  //std::vector<cv::Mat> videoFrames;

  for (int i = 0;;i++)
  {
    cap.set(CV_CAP_PROP_POS_FRAMES,i);
    cap.read(frame);

    inputFrames.push_back(frame);
    if (!frame.data) break;
    frame.release();
  }
  cap.release();
}

void writeVideo()
{

}

int main(int argc,char* argv[])
{
  if (argc <  1) 
  { 
    std::cout << "Usage: main <video file>\n"; 
    return 0;
  }

  std::vector<cv::Mat> videoFrames;
  
  readVideo(videoFrames,argv[1]);
  interpolate(videoFrames);
  //test(videoFrames);

  //Wait until any key is pressed
  cv::waitKey(0);
  std::cout << "Exiting application\n";

  return 0;
}

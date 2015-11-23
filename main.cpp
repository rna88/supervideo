#include </usr/include/opencv/cv.h>
#include </usr/include/opencv/ml.h>
#include </usr/include/opencv/cxcore.h>
#include </usr/include/opencv/highgui.h>
 
using namespace cv;
//using namespace std;

void test(std::vector<cv::Mat>& frame)
{
  imshow("FrameFirst",frame[0]); 
  imshow("FrameSecond",frame[250]); 
  imshow("Difference",frame[1] - frame[250]);  
}

void interpolate(/*args*/)
{

}

int main(int argc,char* argv[])
{
  cv::VideoCapture cap;
  //cap.open("bird.avi");
  cap.open(argv[1]);
  if (!cap.isOpened()) std::cout << "FAILED TO OPEN VIDEO.";
  
  //for (int i = 0; i < 
  
  cv::Mat frame;
  std::vector<cv::Mat> videoFrame;

  for (int i = 0;;i++)
  {
    cap.set(CV_CAP_PROP_POS_FRAMES,i);
    cap.read(frame);

    videoFrame.push_back(frame);
    if (!frame.data) break;
    //std::cout << "Frame " << i << " ";
    frame.release();
  }

  test(videoFrame);

  //Wait until any key is pressed
  cv::waitKey(0);
  std::cout << "Exiting application\n";

return 0;
}

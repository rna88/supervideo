#include </usr/include/opencv/cv.h>
#include </usr/include/opencv/ml.h>
#include </usr/include/opencv/cxcore.h>
#include </usr/include/opencv/highgui.h>
 
using namespace cv;
//using namespace std;

#define THRESH_VALUE 200
#define THRESH_MODE 0
#define THRESH_MAX_VALUE 255
#define PROJ_PATH "/home/pool/Documents/ENSC440/"
 
int main(int args,char* argv[])
{
  cv::VideoCapture cap;
  //cap.open("bird.avi");
  cap.open(argv[1]);
  if (!cap.isOpened()) std::cout << "FAILED TO OPEN VIDEO.";
  
  //for (int i = 0; i < 
  
  cv::Mat frame;
  cv::Mat thresh_img;
  cap >> frame;
  
  cv::cvtColor( frame, thresh_img, CV_RGB2GRAY );
  cv::threshold( thresh_img, thresh_img, THRESH_VALUE, THRESH_MAX_VALUE, THRESH_MODE );
  //cv::Mat threshVideo;;
		
  //lets show the output
  cv::imshow("original", frame);
  cv::imshow("Thresholded", thresh_img);
                                    
    //Wait until any key is pressed
  cv::waitKey(0);
  std::cout << "Exiting application\n";

return 0;
}

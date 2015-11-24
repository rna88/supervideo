#include </usr/include/opencv/cv.h>
#include </usr/include/opencv/ml.h>
#include </usr/include/opencv/cxcore.h>
#include </usr/include/opencv/highgui.h>
 
//using namespace cv;
//using namespace std;

class supervideo
{
public:

  struct channels { cv::Mat Y,U,V; };

  supervideo()
  {
  }

  ~supervideo()
  {
  }

  void readVideo(char* fileName)
  {
    capture.open(fileName);
    if (!capture.isOpened()) std::cout << "FAILED TO OPEN VIDEO.";
    
    cv::Mat frame;
    //std::vector<cv::Mat> videoFrames;

    for (int i = 0;;i++)
    {
//      capture.set(CV_CAP_PROP_POS_FRAMES,i);
      capture.read(frame);
      if (!frame.data) break;
      inFrames.push_back(frame);
      frame.release();
//      std::cout << i << " ";
    }
  }

  void interpolate()
  {
    channels frame = convertToYUV(inFrames[0]);
    imshow("Y", frame.Y);
    imshow("Cr", frame.U);
    imshow("Cb", frame.V);
  }

  void resize(char* scaleFactorString)
  {
    double scaleFactor = std::atof(scaleFactorString);
    cv::Size outImageSize(0,0);

    //cv::resize(inFrames[0],inFrames[0],outImageSize,scaleFactor,scaleFactor,cv::INTER_CUBIC);
    //imshow("Resized",inFrames[0]);

    std::cout << "Total # Frames: " << inFrames.size() << std::endl;

    for (unsigned long int i = 0; i < inFrames.size(); i++)
    {
      std::cout << "\r" << "Scaling Frame: " << i+1 << "/" << inFrames.size();
      cv::resize(inFrames[i],inFrames[i],outImageSize,scaleFactor,scaleFactor,cv::INTER_CUBIC);
      //outFrames.push_back(inFrames[i]);
      //std::cout << i << " " << inFrames[i].rows << " ";;
    }
    std::cout << "\n";
  }
  
  void writeVideo(char* outFileName)
  {
    // These parameters need to be changed to reflect any previous processing.
    double inputFPS = capture.get(CV_CAP_PROP_FPS);
    int width = inFrames[0].cols; 
    int height = inFrames[0].rows; 
  
    cv::Size frameSize(width, height);
    cv::VideoWriter writer(outFileName, CV_FOURCC('D','I','V','3'), inputFPS, frameSize, true);
  
    if (writer.isOpened())
    {
      for (unsigned long int i = 0; i < inFrames.size();i++)
      {
        std::cout << "\r" << "Writing Frame: " << i+1 << "/" << inFrames.size();
        writer.write(inFrames[i]);
      }
      std::cout << "\n";
    }
    else 
    {
      std::cout << "Video failed to write";
      exit(1);
    }
  }

private:

  std::vector<cv::Mat> inFrames;
  std::vector<cv::Mat> outFrames;
  cv::VideoCapture capture;

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

};

int main(int argc,char* argv[])
{
  if (argc <= 1) 
  { 
    std::cout << "Usage: main <input video file> <output video file>\n"; 
    exit(1);
  }

  supervideo sv;
  sv.readVideo(argv[1]);
//  sv.interpolate();
  sv.resize(argv[3]);
  sv.writeVideo(argv[2]);

  //Wait until any key is pressed
  cv::waitKey(0);
  std::cout << "Exiting application\n";

  return 0;
}

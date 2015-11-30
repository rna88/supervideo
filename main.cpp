#include </usr/include/opencv2/opencv.hpp>

#define Y 0

class supervideo
{
public:

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
    //channels frame = convertToYUV(inFrames[0]);
    //imshow("Y", frame.Y);
    //imshow("Cr", frame.U);
    //imshow("Cb", frame.V);
  }

  void resize(char* scaleFactorString)
  {
    scaleFactor = std::atof(scaleFactorString);
    cv::Size outImageSize(0,0);

    std::cout << "Total # Frames: " << inFrames.size() << std::endl;

    for (unsigned long int i = 0; i < inFrames.size(); i++)
    {
      std::cout << "\r" << "Scaling Frame: " << i+1 << "/" << inFrames.size();
      outFrames.push_back(inFrames[i]);
      cv::resize(outFrames[i],outFrames[i],outImageSize,scaleFactor,scaleFactor,CV_INTER_CUBIC);
      //std::cout << i << " " << inFrames[i].rows << " ";;
    }
    std::cout << "\n";
    
    //cv::Mat grayFrame;
    //cv::cvtColor(outFrames[0],grayFrame,CV_RGB2YCrCb);
    //imshow("YUV",grayFrame); 

    ////cv::Scalar gpValue = grayFrame.at<uchar>(0,0);

    ////Scalar 
    //cv::cvtColor(grayFrame,grayFrame,CV_YCrCb2RGB);
    //imshow("RGB",grayFrame); 

    //channels YUVMat = convertToYUV(outFrames[0]);

    //std::vector<cv::Mat> YUVMat = convertToYUV(outFrames[0]);
    //imshow("Y",YUVMat[Y]);  
    
    cv::Mat YUV;
    cv::cvtColor(outFrames[0],YUV,CV_RGB2YCrCb);

    imshow("Y",YUV);
    
    
    //std::cout << inFrames[0] << std::endl;
    
    //std::cout << (int)YChannel << std::endl;
    cv::Mat RGB;
    cv::cvtColor(YUV,RGB,CV_YCrCb2RGB);
    imshow("R",RGB);


    for (int y = 0; y < YUV.cols; y++)
    {
      for (int x = 0; x < YUV.rows; x++)
      {
	setValue(YUV,x,y,0);
      }
    //  std::cout << "\n";
    }

//    for (int y = 0; y < YUV.cols; y++)
//    {
//      for (int x = 0; x < YUV.rows; x++)
//      {
//        cv::Vec3b image = YUV.at<cv::Vec3b>(y,x);
//        //std::cout << (int)image.val[Y] << "|";
//	//std::cout << (int)image.val[Y]  << "|";
//      }
//      std::cout << "\n";
//    }

    cv::Mat RGB_mod;
    cv::cvtColor(YUV,RGB_mod,CV_YCrCb2RGB);
    imshow("R_mod",RGB_mod);


   // imshow("Changed",YUVMat[Y]);  
    //cv::Mat RGBMat = convertToRGB(YUVMat);
    //imshow("out",RGBMat);

    // apply ALD operation (local sharp edge detector).
    //channels ALDResult = convertToYUV(inFrames[0]);
    
    float searchRadius = 5; // 5 pixels.
    //cv::Mat grayFrame  = ALDResult.Y;
    //cv::Mat extractedEdges;

    //extractedEdges.rows = grayFrame.rows;
    //extractedEdges.cols = grayFrame.cols;

//  //  cv::cvtColor(inFrames[0],grayFrame,CV_RGB2GRAY);

    //float currentALD = 0.0;
    //for (int y = 0; y < grayFrame.cols; y++)
    //{
    //  for (int x = 0; x < grayFrame.rows; x++)
    //  {
    //    calculateALD(grayFrame, cv::Point(x,y), searchRadius);
    //     
    //  }
    //}
  }
  
  void writeVideo(char* outFileName)
  {
    // These parameters need to be changed to reflect any previous processing.
    double inputFPS = capture.get(CV_CAP_PROP_FPS);
    int width = outFrames[0].cols; 
    int height = outFrames[0].rows; 
  
    cv::Size frameSize(width, height);
    cv::VideoWriter writer(outFileName, CV_FOURCC('D','I','V','3'), inputFPS, frameSize, true);
  
    if (writer.isOpened())
    {
      for (unsigned long int i = 0; i < outFrames.size();i++)
      {
        std::cout << "\r" << "Writing Frame: " << i+1 << "/" << outFrames.size();
        writer.write(outFrames[i]);
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

  double scaleFactor;
  std::vector<cv::Mat> inFrames;
  std::vector<cv::Mat> outFrames;
  cv::VideoCapture capture;

  void setValue(cv::Mat& img,int x, int y, int value)
  {
    cv::Vec3b image = img.at<cv::Vec3b>(x,y);
    image.val[Y] = value;
    img.at<cv::Vec3b>(x,y) = image;
  }

  std::vector<cv::Mat> convertToYUV(cv::Mat image)
  {
    cv::cvtColor(image, image, CV_RGB2YCrCb);
    std::vector<cv::Mat> channel;
    cv::split(image, channel);
    return channel;
  }

  cv::Mat convertToRGB(std::vector<cv::Mat> input)
  {
    cv::Mat output; 
    cv::merge(input,output);
    cv::cvtColor(output,output,CV_YCrCb2RGB);
    return output;
  }

  float distance(float x1, float y1, float x2, float y2)
  {

    return cv::sqrt( (x1-x2)*(x1-x2) - (y1-y2)*(y1-y2) );
  }

  float calculateALD(cv::Mat& inImage, cv::Point gp, float radius)
  {
    float ALD = 0;
    int p = 0;
    cv::Scalar gcValue = 0;
    cv::Scalar gpValue = 0;
    
    // Limit search to valid areas on the image.
    for (int y = (gp.y - radius); y < inImage.cols && y >= 0; y++)
    {
      for (int x = (gp.x - radius); x < inImage.rows && x >= 0; x++)
      {
	if ( distance(gp.x, gp.y, x, y) <= radius && x >= 0 && y >= 0 )
	{
       	  p++;
	  gpValue = inImage.at<double>(gp.y,gp.x);
	  gcValue = inImage.at<double>(y,x);
          ALD += cv::fast_abs(gpValue.val[0] - gcValue.val[0]);
	}
      }
    }

    ALD /= float(p);

    return ALD;
  }
};

int main(int argc,char* argv[])
{
  if (argc <= 1) 
  { 
    std::cout << "Usage: supervideo <input video file> <output video file> <scale Factor>\n"; 
    return 0 /*exit(1)*/;
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

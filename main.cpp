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


    cv::Mat YUVOut = outFrames[0];
    cv::imshow("OriginalRGB",YUVOut);
    std::vector<cv::Mat> YUVChannels = convertToYUV(YUVOut);
   
    cv::imshow("YUVOut",YUVOut);

    cv::Mat YChannel = YUVChannels[Y];
    cv::imshow("Ychannel", YChannel);
    // Use YChannel for algorthm. 


    //cv::Mat YUV;
    //cv::Mat YUV2;
    //cv::cvtColor(outFrames[0],YUV,CV_RGB2YCrCb);
    //cv::cvtColor(outFrames[0],YUV2,CV_RGB2YCrCb);
    //imshow("Y",YUV);
    
    //std::cout << inFrames[0] << std::endl;
    
    //std::cout << (int)YChannel << std::endl;
    //cv::Mat RGB;
    //cv::cvtColor(YUV,RGB,CV_YCrCb2RGB);
    //imshow("R",RGB);

    //cv::Mat gray = RGB;
    //cv::cvtColor(gray,gray,CV_RGB2GRAY);


   // for (int y = 0; y < YUV.cols; y++)
   // {
   //   for (int x = 0; x < YUV.rows; x++)
   //   {
   //     int ct = readValue(YUV,x,y) + 10;
   //     setValue(YUV,x,y,ct);
   //   }
   // //  std::cout << "\n";
   // }

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

   // cv::Mat RGB_mod;
   // cv::cvtColor(YUV,RGB_mod,CV_YCrCb2RGB);
   // imshow("R_mod",RGB_mod);


    // imshow("Changed",YUVMat[Y]);  
    //cv::Mat RGBMat = convertToRGB(YUVMat);
    //imshow("out",RGBMat);

    // apply ALD operation (local sharp edge detector).
    //channels ALDResult = convertToYUV(inFrames[0]);
    
    float searchRadius = 10; // 5 pixels.
    //cv::Mat grayFrame  = ALDResult.Y;
    //cv::Mat extractedEdges;

    //extractedEdges.rows = grayFrame.rows;
    //extractedEdges.cols = grayFrame.cols;

//  //  cv::cvtColor(inFrames[0],grayFrame,CV_RGB2GRAY);

 //   cv::Mat ALD = YUV;
    //cv::extractChannel(YUV,ALD,0);
    //imshow("ALD",ALD);
    
    //for (int y = 0; y < ALD.rows;y++)
    //{
    //  //for (int x = 0; x < ALD.rows;x++)
    //  //{
    //    //setValue(ALD,y,x,0); 
    //          std::cout << ALD.at<uchar>(y,0) << ";";
    //  //}
    //  //std::cout << "\n";
    //}

    
    //std::cout << calculateALD(YUV, cv::Point(330,1), searchRadius) << std::endl;

//    for (int y = 0; y < YUV.rows; y++)
//    {
//      for (int x = 0; x < YUV.cols; x++)
//      {
//        setPixel(YUV,x,y,(int)calculateALD(YUV, cv::Point(x,y), searchRadius));
//      }
//    }

    
    //std::cout << YUV.cols << "\n";

    //cv::Mat eroded = YUV;
    ////erode(YUV, eroded,3,3); 
    //cv::extractChannel(eroded,eroded,0);
    //cv::imshow("eroded",eroded);

    //cv::Mat ALD;
    //cv::extractChannel(YUV,ALD,0);
    //cv::imshow("ALD",ALD);

// Change to only handle 1 channel
    //std::cout << (int)ALD.at<unsigned char>(20,40) << "safsdf";
    //std::cout << "LOL\n\n\n\n";
   
    for (int y = 0; y < YChannel.rows; y++)
    {
      for (int x = 0; x < YChannel.cols; x++)
      {
        //setPixel(YUV,x,y,(int)calculateALD(YUV, cv::Point(x,y), searchRadius));
        
        YChannel.at<unsigned char>(y,x) = (int)calculateALD(YChannel, cv::Point(x,y), searchRadius);
        //std::cout << (int)ALD.at<unsigned char>(y,x) << ":";
      }
    }


    //cv::imshow("Ych",YChannel);
    //cv::imshow("orig",outFrames[0]);
    
    //Assign modified Y channel back to vector
    YUVChannels[Y] = YChannel;  
    cv::imshow("ALD",YChannel);

    // convert back to RGB format.
    cv::Mat RGB = convertToRGB(YUVChannels);
    cv::imshow("RGB",RGB);

    
    //cv::imshow("ALD2",ALD);
    //cv::Mat Ychannel;
    //cv::extractChannel(YUV2,Ychannel,0);
    //cv::imshow("Y",Ychannel);
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

  void testPSNR()
  {
    double avgPSNR = 0;
    cv::Size outImageSize(0,0);

    for (unsigned long int i = 0; i < inFrames.size(); i++ ) 
    {
      cv::resize(outFrames[i],outFrames[i],outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_NN);
      avgPSNR += getPSNR(inFrames[i],outFrames[i]);
    }
    avgPSNR /= inFrames.size();

    std::cout << "Average PSNR per frame: " << avgPSNR << std::endl;
  }

private:

  double scaleFactor;
  std::vector<cv::Mat> inFrames;
  std::vector<cv::Mat> outFrames;
  cv::VideoCapture capture;

  int readPixel(cv::Mat& img,int x, int y)
  {
    cv::Vec3b image = img.at<cv::Vec3b>(y,x);
    int value = image.val[Y];
    return value;
  }

  void setPixel(cv::Mat& img,int x, int y, int value)
  {
    cv::Vec3b image = img.at<cv::Vec3b>(y,x);
    image.val[Y] = value;
    img.at<cv::Vec3b>(y,x) = image;
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

  float calculateALD(cv::Mat& inImage, cv::Point gc, float radius)
  {
    float ALD = 0;
    int p = 0;
    int gpValue = 0;
    int gcValue = (int)inImage.at<unsigned char>(gc.y,gc.x);
    //int gcValue = readPixel(inImage, gc.x, gc.y);
    
    // Limit search to valid areas on the image.
    for (int y = (gc.y - radius); y <= (gc.y + radius); y++)
    {
      for (int x = (gc.x - radius); x <= (gc.x + radius); x++)
      {
	if ( (distance(gc.x, gc.y, x, y) <= radius) && 
			(x >= 0) && 
			(y >= 0) &&
			(x < inImage.cols) &&
			(y < inImage.rows) )
	{
       	  p++;
	  gpValue = (int)inImage.at<unsigned char>(y,x);
	  //gpValue = readPixel(inImage, x, y);
          ALD += cv::fast_abs( gpValue - gcValue );
	  //if (x == 350) std:: cout << "hey";
	}
      }
    }

    ALD /= float(p);

    //ALD=1.0;
    return ALD;
  }

  int findMin(cv::Mat& img,cv::Point p,int s,int t)
  {
    int min = 255;

    for (int y = (p.y - t); y <= (p.y + t); y++)
    {
      for (int x = (p.x - s); x <= (p.x + s); x++)
      {
        int currentPixel = readPixel(img,x,y);
        if ( currentPixel < min  && 
			(x >= 0) &&
			(y >= 0) &&
			(x < img.cols) &&
			(y < img.rows)) 
	{
          min = currentPixel;
	}
      }
    }
	//std::cout << min << ":";
    return min;
  }

  // s,t indicate length/2 of area to search for min. eg s,t=1 -> 3x3 area.
  void erode(cv::Mat& img, cv::Mat& out, int s, int t)
  {
    for (int y = 0; y < img.rows;y++) 
    {
      for (int x = 0; x < img.cols;x++) 
      {
        setPixel(out,x,y,findMin(img,cv::Point(x,y),s,t)); 
      }
    }
  } 

  // Reference this.
  double getPSNR(const cv::Mat& I1, const cv::Mat& I2)
  {
    cv::Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    cv::Scalar s = sum(s1);        // sum elements per channel
    
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    
    if( sse <= 1e-10) // for small values return zero
      return 0;
    else
    {
      double mse  = sse / (double)(I1.channels() * I1.total());
      double psnr = 10.0 * log10((255 * 255) / mse);
      return psnr;
    }
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
  sv.testPSNR();


  //Wait until any key is pressed
  cv::waitKey(0);
  std::cout << "Exiting application\n";

  return 0;
}

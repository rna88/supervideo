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
    // Use YChannel for rest of algorthm. 

    cv::Mat ALD = getALD(YChannel,15);
    cv::imshow("ALD",ALD);
    //YChannel = ALD;
    
    //float searchRadius = 10; // 5 pixels.
  
    //for (int y = 0; y < YChannel.rows; y++)
    //{
    //  for (int x = 0; x < YChannel.cols; x++)
    //  {
    //    //setPixel(YUV,x,y,(int)calculateALD(YUV, cv::Point(x,y), searchRadius));
    //    YChannel.at<unsigned char>(y,x) = (int)calculateALD(YChannel, cv::Point(x,y), searchRadius);
    //    //std::cout << (int)ALD.at<unsigned char>(y,x) << ":";
    //  }
    //}
    
    //cv::Mat canny=YChannel;
    //cv::Canny(YChannel,canny,50,90);
    //cv::imshow("canny",outFrames[0]);

    // Extract edges from ALD.
    cv::Mat extractMask;
    cv::Mat extractedEdges;
    cv::adaptiveThreshold(ALD, extractMask, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 3);
    cv::bitwise_not( extractMask, extractMask );
//    cv::GaussianBlur(extractMask, extractMask, cv::Size(3,3),5);
    cv::imshow("extractMask",extractMask);
    YChannel.copyTo(extractedEdges,extractMask);
    cv::imshow("extractedEdges",extractedEdges);

    // Resize image and blur it.
    cv::Size imageSize(0,0);
    cv::resize(extractedEdges,extractedEdges,imageSize,scaleFactor,scaleFactor,CV_INTER_LINEAR);
    cv::GaussianBlur(extractedEdges, extractedEdges, cv::Size(3,3),5);
    cv::imshow("extractedEdgesx2",extractedEdges);

    // Apply erosion operator.
    cv::Mat erodedExtractedEdges;
    cv::erode(extractedEdges,erodedExtractedEdges,cv::Mat());
    cv::imshow("erodedExtractedEdges",erodedExtractedEdges);

    // Sharpen and downsample. 
    cv::Mat tmp;
    cv::GaussianBlur(erodedExtractedEdges, tmp, cv::Size(5,5), 5);
    cv::addWeighted(erodedExtractedEdges, 1.5, tmp, -0.5, 0, erodedExtractedEdges);
    cv::imshow("erodedEE_sharp", erodedExtractedEdges);
	   
    cv::resize(erodedExtractedEdges,erodedExtractedEdges,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_LINEAR);
    cv::imshow("erodedEE_sharp_downsized", erodedExtractedEdges);

    erodedExtractedEdges*=2;
    cv::imshow("erodedEE_sharp_downsized * 2", erodedExtractedEdges);

    //cv::Mat SobelGrad = YChannel;

    //cv::Mat grad = extractedEdges;
    //int scale = 1;
    //int delta = 0;
    //int ddepth = CV_16U;
    //cv::Mat sx = extractedEdges;
    //cv::Mat sy = extractedEdges; 
    //cv::Mat abs_sx = extractedEdges;
    //cv::Mat abs_sy = extractedEdges; 

    //cv::GaussianBlur( ALD, SobelGrad, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    //cv::Sobel(SobelGrad,sx,ddepth,1,0,3,scale,delta,cv::BORDER_DEFAULT);
    //cv::Sobel(SobelGrad,sy,ddepth,0,1,3,scale,delta,cv::BORDER_DEFAULT);
    //cv::convertScaleAbs( sx, abs_sx );
    //cv::convertScaleAbs( sy, abs_sy );
    //cv::addWeighted( abs_sx, 0.5, abs_sy, 0.5, 0, grad );
    //cv::imshow("gradientMap",grad);
    //cv::imshow("sx",sx);
    //cv::imshow("sy",sy);
    
    //cv::Mat gradientMap = getGradient(extractedEdges);
    //cv::imshow("gradientMap",gradientMap);

    // Create texture image.
    //cv::Mat texture = ALD - sharpenFilter /*.inv(CV_B)*/;
    //cv::imshow("texture",texture);
    
    //implement HR gradient smoothing.

    //sharpen
    //cv::Mat tmp;
    //cv::GaussianBlur(grad, tmp, cv::Size(5,5), 5);
    //cv::addWeighted(grad, 1.5, tmp, -0.5, 0, grad);
    //cv::imshow("gradientMap_sharp",grad);
	
//  int kernel_size = 3;
//    int scale = 1;
//      int delta = 0;
//        int ddepth = CV_16S;
////
//	cv::Mat dst,abs_dst;
////	//cv::GaussianBlur( extractedEdges, extractedEdges, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
//	cv::Laplacian( extractedEdges, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT );
//	cv::convertScaleAbs( dst, abs_dst );
//	cv::GaussianBlur( abs_dst, abs_dst, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
//	cv::imshow("Lapacian",abs_dst);


    //Assign modified Y channel back to vector
    YUVChannels[Y] = YChannel;  

    // convert back to RGB format.
    cv::Mat RGB = convertToRGB(YUVChannels);
    cv::imshow("RGB",RGB);
    //outFrames[i] = RGB;
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

  int calculateGrad(cv::Mat img, cv::Point p)
  {
    // TODO: Fix gradient calc so that edge pixels have non-zero value.
    int gradMagnitude = 0;
    int xgrad = 0;
    int ygrad = 0;
//    if ( (p.x - 1 >= 0) &&
//	 (p.y - 1 >= 0) &&
//	 (p.x + 1 < img.cols) &&
//	 (p.y + 1 < img.rows) )
    {
      // do xy gradient 
      //xgrad = ((int)img.at<unsigned char>(p.x + 1, p.y)) - ((int)img.at<unsigned char>(p.x - 1, p.y)) ;
      //ygrad = ((int)img.at<unsigned char>(p.x, p.y + 1)) - ((int)img.at<unsigned char>(p.x, p.y - 1)) ;

      //xgrad = ((int)img.at<unsigned char>(p.y , p.x + 1)) - ((int)img.at<unsigned char>( p.y , p.x - 1)) ;
      //ygrad = ((int)img.at<unsigned char>(p.y + 1 , p.x )) - ((int)img.at<unsigned char>(p.y - 1 , p.x)) ;
      
      xgrad = ((int)img.at<unsigned char>(p.y , p.x + 1)) - ((int)img.at<unsigned char>(p.y , p.x - 1)) ;
      ygrad = ((int)img.at<unsigned char>(p.y + 1 , p.x )) - ((int)img.at<unsigned char>(p.y - 1 , p.x)) ;


      gradMagnitude = cv::sqrt( (xgrad*xgrad) + (ygrad*ygrad) ); 
      //std::cout << xgrad << "," << ygrad << ":";
     // std::cout << gradMagnitude << ":";

      //cv::magnitude(xgrad,ygrad,gradMagnitude); 
      //gradMagnitude = xgrad;
    }
    return gradMagnitude; 
  }

  cv::Mat getGradient(cv::Mat img)
  {
    cv::Mat grad = img;

    for (int y = 0; y < img.rows; y++) 
    {
      for (int x = 0; x < img.cols; x++) 
      {
	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
	grad.at<unsigned char>(y,x) = calculateGrad(img, cv::Point(x,y));
      }
    }

    return grad;
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

  cv::Mat getALD(cv::Mat& imgY, int radius)
  {
    cv::Mat ALD = imgY;

    for (int y = 0; y < ALD.rows; y++)
    {
      for (int x = 0; x < ALD.cols; x++)
      {
        ALD.at<unsigned char>(y,x) = (int)calculateALD(ALD, cv::Point(x,y), radius);
      }
    }
    return ALD;
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

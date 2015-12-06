#include </usr/include/opencv2/opencv.hpp>
#include <bitset>

#define Y 0
#define SHOW_IMAGES 1
//const char* input = "YChannel.png";
const char* input = "lena.jpg";
const int ALDRadius = 50;

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
      orgFrames.push_back(frame);
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

    for (unsigned long int i = 0; i < inFrames.size()/100; i++)
    {
      std::cout << "\r" << "Scaling Frame: " << i+1 << "/" << inFrames.size();
      outFrames.push_back(inFrames[i]);
      cv::resize(outFrames[i],outFrames[i],outImageSize,scaleFactor,scaleFactor,CV_INTER_CUBIC);
      //std::cout << i << " " << inFrames[i].rows << " ";;
      sharpenEdges(i);
//      finalTexture(

    }
    std::cout << "\n";
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

  void testSharpenEdges(char* scaleFactorString)
  {
    scaleFactor = std::atof(scaleFactorString);
    cv::Size outImageSize(0,0);

// grab original Y channel.
    cv::Mat inputImage = cv::imread(input);
    if (inputImage.rows == 0) std::cout << "NOPE\n";
    
    // downscale original so we can test the uprezzed
    cv::Mat inputDownscaled;
    cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    //cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);


    cv::Mat inputBICUBIC;
    cv::resize(inputDownscaled,inputBICUBIC,outImageSize,scaleFactor,scaleFactor,CV_INTER_CUBIC);

   // cv::Mat YUVInBi = inputDownscaled.clone();
   // std::vector<cv::Mat> originalYUVChannels = convertToYUV(YUVIn);
   // cv::Mat originalYChannel = originalYUVChannels[Y];
   // // input bicubic is real resized bicubic?????????????????



    cv::Mat YUVIn = inputDownscaled.clone();
    std::vector<cv::Mat> originalYUVChannels = convertToYUV(YUVIn);
    cv::Mat originalYChannel = originalYUVChannels[Y];
    // input bicubic is real resized bicubic?????????????????

   
    // grab bicubic resized Y channel.
    cv::Mat YUVOut = inputBICUBIC.clone();
#if SHOW_IMAGES
    cv::imshow("OriginalRGB",YUVOut);
#endif
    std::vector<cv::Mat> YUVChannels = convertToYUV(YUVOut);
    //cv::imshow("YUVOut",YUVOut);

    cv::Mat YChannel = YUVChannels[Y];
#if SHOW_IMAGES
    cv::imshow("Ychannel", YChannel);
#endif
    // Use YChannel for rest of algorthm. 


    // Calculate ALD.
    cv::Mat YChannelCopy = YChannel.clone();
    cv::Mat ALD = getALD(YChannelCopy,ALDRadius);
#if SHOW_IMAGES
    cv::imshow("ALD",ALD);
#endif
    //YChannel = ALD;
    

    cv::Mat gradient_bicubic = getGradient(ALD.clone());
#if SHOW_IMAGES
    cv::imshow("Gradient",gradient_bicubic);
#endif
    // G_b = gradient_bicubic.



    cv::Size blurKernel = cv::Size(3,3);

    // Extract edges from ALD.
    cv::Mat extractMask = ALD.clone();
    cv::Mat extractedEdges;
    //
    cv::Scalar threshold = cv::mean(ALD);
    float avgALD = 1.0 * threshold.val[0];
    //cv::threshold(ALD,extractMask,avgALD, 0, CV_THRESH_TOZERO); 

    cv::Mat gradient_b_dilated;
    cv::dilate(gradient_bicubic,gradient_b_dilated,cv::Mat());
    cv::imshow("gbd",gradient_b_dilated);

   for (int y = 0; y < gradient_bicubic.rows; y++) 
    {
      for (int x = 0; x < gradient_bicubic.cols; x++) 
      {
	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
	if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  10)
	{
          extractMask.at<unsigned char>(y,x) = 255;
	}
	else 
	{
          extractMask.at<unsigned char>(y,x) = 0;
	}
      }
    }
    //for (int y = 0; y < ALD.rows; y++) 
    //{
    //  for (int x = 0; x < ALD.cols; x++) 
    //  {
    //    //grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
    //    if ((int)ALD.at<unsigned char>(y,x) >  .8*avgALD && (int)ALD.at<unsigned char>(y,x) < 1.2*avgALD)
    //    {
    //      extractMask.at<unsigned char>(y,x) = 255;
    //    }
    //    else 
    //    {
    //      extractMask.at<unsigned char>(y,x) = 0;
    //    }
    //  }
    //}	    
    //cv::adaptiveThreshold(ALD, extractMask, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 4);
    //cv::bitwise_not( extractMask, extractMask );
#if SHOW_IMAGES
    cv::imshow("extractMask",extractMask);
#endif
    //YChannel.copyTo(extractedEdges,extractMask);
    //gradient_bicubic.copyTo(extractedEdges,extractMask);   //////////////// is this the proper way to mask?
 // cv::multiply(extractedEdges,extractMask,extractedEdges);


// standin code - sharpens bicubic gradient map.
      gradient_bicubic = getGradient(YChannel.clone());
      cv::Mat blurred; double sigma = 1, s_threshold = 5, amount = 1;
      GaussianBlur(gradient_bicubic, blurred, cv::Size(), sigma, sigma);
      cv::Mat lowContrastMask = abs(gradient_bicubic - blurred) < s_threshold;
      cv::Mat sharpened = gradient_bicubic*(1+amount) + blurred*(-amount);
      gradient_bicubic.copyTo(sharpened, lowContrastMask);
  cv::imshow("sdf",sharpened);

    extractedEdges = getGradient(YChannel);
   // gradient_bicubic.copyTo(extractedEdges,extractMask);   //////////////// is this the proper way to mask?

    //extractedEdges=gradient_bicubic.clone();
#if HOW_IMAGES
    cv::imshow("extractedEdges",extractedEdges);
#endif

    // Resize image and blur it.
    cv::Size imageSize(0,0);
    cv::resize(extractedEdges,extractedEdges,imageSize,scaleFactor,scaleFactor,CV_INTER_NN);
    //cv::blur(extractedEdges, extractedEdges, blurKernel);
    cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
    //extractedEdges = sharpen(extractedEdges);
   // extractedEdges = sharpen(extractedEdges);
#if SHOW_IMAGES
    cv::imshow("extractedEdgesx2",extractedEdges);
#endif

    // Apply erosion operator.
    cv::Mat erodedExtractedEdges;
    cv::erode(extractedEdges,erodedExtractedEdges,cv::Mat(),cv::Point(-1,-1),1);
#if SHOW_IMAGES
    cv::imshow("erodedExtractedEdges",erodedExtractedEdges);
#endif

    // Reblur and downsample. 
    //cv::blur(erodedExtractedEdges, erodedExtractedEdges, blurKernel);
    cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
    //erodedExtractedEdges = sharpen(erodedExtractedEdges);
#if SHOW_IMAGES
    cv::imshow("erodedEE_sharp", erodedExtractedEdges);
#endif
	   
    cv::resize(erodedExtractedEdges,erodedExtractedEdges,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_CUBIC);
#if SHOW_IMAGES
    cv::imshow("erodedEE_sharp_downsized", erodedExtractedEdges);
#endif

   // erodedExtractedEdges *= 2;  
#if SHOW_IMAGES
    cv::imshow("erodedEE_sharp_downsized * 2", erodedExtractedEdges);
#endif
    // G_hat = erodedExtractedEdges.



//    cv::Mat HREDownsampled;
//    cv::blur(YChannel, HREDownsampled, blurKernel);
//    cv::resize(HREDownsampled,HREDownsampled,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_NN);
//    cv::imshow("extracted_edges_blurred",HREDownsampled);
//    
//    //cv::Mat originalDiff = originalYChannel - HREDownsampled;
//    cv::Mat originalDiff = originalYChannel - HREDownsampled;
//
//    // Upscale and reblur.
//    cv::resize(originalDiff,originalDiff,imageSize,scaleFactor,scaleFactor,CV_INTER_NN);
//    cv::blur(originalDiff, originalDiff, blurKernel);
//    cv::imshow("Difference with original",originalDiff);
//    //originalDiff *= 0.2;


  // standin code - sharpens bicubic gradient map.
      //gradient_bicubic = getGradient(YChannel.clone());
      //cv::Mat blurred; double sigma = 1, s_threshold = 5, amount = 1;
      //GaussianBlur(gradient_bicubic, blurred, cv::Size(), sigma, sigma);
      //cv::Mat lowContrastMask = abs(gradient_bicubic - blurred) < s_threshold;
      //cv::Mat sharpened = gradient_bicubic*(1+amount) + blurred*(-amount);
      //gradient_bicubic.copyTo(sharpened, lowContrastMask);
#if SHOW_IMAGES
      //cv::imshow("gb sharp", sharpened);
      //cv::imshow("low contrast", lowContrastMask);
#endif
      //gradient_bicubic = sharpened;

    //pow(gradient_bicubic,2,gradient_bicubic);
    pow(erodedExtractedEdges,2,erodedExtractedEdges);
#if SHOW_IMAGES
    //imshow("Gradient - GradientMap",erodedExtractedEdges - gradient_bicubic);
#endif


   // std::vector<cv::Mat> orgY = convertToYUV(inputImage);
   // cv::Mat orgYChannel = orgY[Y].clone();
    
     
    //cv::Mat textureArea = finalTexture (orgYChannel, YChannel, scaleFactor);


    cv::Mat maskedY; 
    cv::Mat downSampledExtractMask;
    cv::resize(extractMask,downSampledExtractMask,cv::Size(0,0),1/scaleFactor,1/scaleFactor,CV_INTER_CUBIC);
    //multiply(originalYChannel,downSampledExtractMask,maskedY);
    //multiply(YChannel,extractMask,YChannel);
    gradient_bicubic = getGradient(YChannel);
    cv::Mat originalDiff;  
    // Iterative formula
    cv::Mat finalEdgeResult;
    int i = 0;
    while (i < 50)
    {
      //cv::Mat originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
      originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
      //originalDiff = getDifference(maskedY, YChannel, blurKernel);
      //cv::Mat orgYresize;
      //cv::resize(originalYChannel,orgYresize,cv::Size(0,0),scaleFactor,scaleFactor);

      pow(gradient_bicubic,2,gradient_bicubic);

      //YChannel = orgYresize;
      YChannel = YChannel + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient_bicubic);
      //YChannel = YChannel + 0.2*originalDiff + 0.004*(sharpened - gradient_bicubic);
      
      //finalEdgeResult = YChannel + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient_bicubic);
      //cv::Mat finalEdgeResult = textureArea + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient_bicubic);
      //cv::Mat finalEdgeResult = originalDiff + YChannel;
      
      gradient_bicubic = getGradient(YChannel);
     //cv::GaussianBlur(gradient_bicubic,gradient_bicubic, cv::Size(3, 3), 3);
      //cv::addWeighted(gradient_bicubic, 1.5, gradient_bicubic, -0.5, 0, gradient_bicubic);
      //cv::imshow("gradient_last_iteration_sharp",gradient_bicubic);

//      cv::pow(gradient_bicubic,2,gradient_bicubic);
//      cv::pow(sharpened,2,sharpened);
//      cv::Mat gradientSubtractionResult;
//
//      cv::subtract(gradient_bicubic,sharpened,gradientSubtractionResult);
//      //cv::absdiff(gradient_bicubic,sharpened,gradientSubtractionResult);
//      cv::imshow("Gradient - GradientMap 2", gradientSubtractionResult);
      i++;
    }
    //cv::imshow("final",finalEdgeResult);
#if SHOW_IMAGES
    cv::imshow("YchannelFinal",YChannel);
      cv::imshow("gradient_last_iteration",gradient_bicubic);
      cv::imshow("originalDiff",originalDiff);
#endif
 
//    std::vector<cv::Mat> orgY = convertToYUV(orgFrames[i]);
//    cv::Mat orgYChannel = orgY[Y].clone();
    
     
//    cv::Mat textureArea = finalTexture (orgYChannel, YChannel, scaleFactor);
    

    //Assign modified Y channel back to vector
    YUVChannels[Y] = YChannel;  
    //YUVChannels[Y] = finalEdgeResult + textureArea;  

    // convert back to RGB format.
    cv::Mat processedRGB = convertToRGB(YUVChannels);
#if SHOW_IMAGES
    cv::imshow("processedRGB",processedRGB);
#endif
    //outFrames[i] = RGB;

    //return finalEdgeResult;
    


    std::cout << "bicubic PSNR: " << getPSNR(inputImage,inputBICUBIC) << "\n";
    std::cout << "sharpened PSNR: " << getPSNR(inputImage,processedRGB) << "\n";
    std::cout << "org PSNR: " << getPSNR(inputImage,inputBICUBIC) << "\n";
  }

  void testSharpenEdges2(char* scaleFactorString)
  {
    scaleFactor = std::atof(scaleFactorString);
    cv::Size outImageSize(0,0);

// grab original Y channel.
    cv::Mat inputImage = cv::imread(input);
    if (inputImage.rows == 0) std::cout << "NOPE\n";
    
    // downscale original so we can test the uprezzed
    cv::Mat inputDownscaled;
    cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    //cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);

    cv::Mat inputBICUBIC;
    cv::resize(inputDownscaled,inputBICUBIC,outImageSize,scaleFactor,scaleFactor,CV_INTER_CUBIC);


    cv::Mat YUVIn = inputDownscaled.clone();
    std::vector<cv::Mat> originalYUVChannels = convertToYUV(YUVIn);
    cv::Mat originalYChannel = originalYUVChannels[Y];
    // input bicubic is real resized bicubic?????????????????

   
    // grab bicubic resized Y channel.
    cv::Mat YUVOut = inputBICUBIC.clone();
#if SHOW_IMAGES
    cv::imshow("OriginalRGB",YUVOut);
#endif
    std::vector<cv::Mat> YUVChannels = convertToYUV(YUVOut);
    //cv::imshow("YUVOut",YUVOut);

    cv::Mat YChannel = YUVChannels[Y];
#if SHOW_IMAGES
    cv::imshow("Ychannel", YChannel);
#endif
    // Use YChannel for rest of algorthm. 


    // Calculate ALD.
    cv::Mat YChannelCopy = YChannel.clone();
    cv::Mat ALD = getALD(YChannelCopy,ALDRadius);
#if SHOW_IMAGES
    cv::imshow("ALD",ALD);
#endif
    //YChannel = ALD;
    

    cv::Mat gradient_bicubic = getGradient(ALD.clone());
#if SHOW_IMAGES
    cv::imshow("Gradient",gradient_bicubic);
#endif
    // G_b = gradient_bicubic.


    cv::Size blurKernel = cv::Size(3,3);

    // Extract edges from ALD.
    cv::Mat extractMask = ALD.clone();
    cv::Mat extractedEdges;
    //
    cv::Scalar threshold = cv::mean(ALD);
    float avgALD = 1.0 * threshold.val[0];
    //cv::threshold(ALD,extractMask,avgALD, 0, CV_THRESH_TOZERO); 

    cv::Mat gradient_b_dilated;
    cv::dilate(gradient_bicubic,gradient_b_dilated,cv::Mat());
    cv::imshow("gbd",gradient_b_dilated);

   for (int y = 0; y < gradient_bicubic.rows; y++) 
    {
      for (int x = 0; x < gradient_bicubic.cols; x++) 
      {
	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
	if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  10)
	{
          extractMask.at<unsigned char>(y,x) = 255;
	}
	else 
	{
          extractMask.at<unsigned char>(y,x) = 0;
	}
      }
    }
    //cv::adaptiveThreshold(ALD, extractMask, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 4);
    //cv::bitwise_not( extractMask, extractMask );
#if SHOW_IMAGES
    cv::imshow("extractMask",extractMask);
#endif
    //YChannel.copyTo(extractedEdges,extractMask);
    //gradient_bicubic.copyTo(extractedEdges,extractMask);   //////////////// is this the proper way to mask?
 // cv::multiply(extractedEdges,extractMask,extractedEdges);


    extractedEdges = getGradient(YChannel);
   // gradient_bicubic.copyTo(extractedEdges,extractMask);   //////////////// is this the proper way to mask?

    //extractedEdges=gradient_bicubic.clone();
#if HOW_IMAGES
    cv::imshow("extractedEdges",extractedEdges);
#endif

    // Resize image and blur it.
    cv::Size imageSize(0,0);
    cv::resize(extractedEdges,extractedEdges,imageSize,scaleFactor,scaleFactor,CV_INTER_NN);
    //cv::blur(extractedEdges, extractedEdges, blurKernel);
    cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
    //extractedEdges = sharpen(extractedEdges);
#if SHOW_IMAGES
    cv::imshow("extractedEdgesx2",extractedEdges);
#endif

    // Apply erosion operator.
    cv::Mat erodedExtractedEdges;
    cv::erode(extractedEdges,erodedExtractedEdges,cv::Mat(),cv::Point(-1,-1),1);
#if SHOW_IMAGES
    cv::imshow("erodedExtractedEdges",erodedExtractedEdges);
#endif

    // Reblur and downsample. 
    //cv::blur(erodedExtractedEdges, erodedExtractedEdges, blurKernel);
    cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
    //erodedExtractedEdges = sharpen(erodedExtractedEdges);
#if SHOW_IMAGES
    cv::imshow("erodedEE_sharp", erodedExtractedEdges);
#endif
	   
    cv::resize(erodedExtractedEdges,erodedExtractedEdges,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_CUBIC);
#if SHOW_IMAGES
    cv::imshow("erodedEE_sharp_downsized", erodedExtractedEdges);
#endif

   // erodedExtractedEdges *= 2;  
#if SHOW_IMAGES
    cv::imshow("erodedEE_sharp_downsized * 2", erodedExtractedEdges);
#endif
    // G_hat = erodedExtractedEdges.

#if SHOW_IMAGES
      //cv::imshow("gb sharp", sharpened);
      //cv::imshow("low contrast", lowContrastMask);
#endif
      //gradient_bicubic = sharpened;

    pow(erodedExtractedEdges,2,erodedExtractedEdges);
#if SHOW_IMAGES
    imshow("Gradient - GradientMap",erodedExtractedEdges - gradient_bicubic);
#endif

    cv::Mat maskedY; 
    cv::Mat downSampledExtractMask;
    gradient_bicubic = getGradient(YChannel);
    cv::Mat originalDiff;  
    
    // Iterative formula
    cv::Mat finalEdgeResult;
    int i = 0;
    while (i < 50)
    {
      originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
      pow(gradient_bicubic,2,gradient_bicubic);

      YChannel = YChannel + 0.2*originalDiff;// + 0.004*(erodedExtractedEdges - gradient_bicubic);
      gradient_bicubic = getGradient(YChannel);
      i++;
    }
    //cv::imshow("final",finalEdgeResult);
#if SHOW_IMAGES
    cv::imshow("YchannelFinal",YChannel);
      cv::imshow("gradient_last_iteration",gradient_bicubic);
      cv::imshow("originalDiff",originalDiff);
#endif
 
    //Assign modified Y channel back to vector
    YUVChannels[Y] = YChannel;  
    //YUVChannels[Y] = finalEdgeResult + textureArea;  

    // convert back to RGB format.
    cv::Mat processedRGB = convertToRGB(YUVChannels);
#if SHOW_IMAGES
    cv::imshow("processedRGB",processedRGB);
#endif

    std::cout << "bicubic PSNR: " << getPSNR(inputImage,inputBICUBIC) << "\n";
    std::cout << "sharpened PSNR: " << getPSNR(inputImage,processedRGB) << "\n";
    std::cout << "org PSNR: " << getPSNR(inputImage,inputBICUBIC) << "\n";
  }

  void testPSNR()
  {
    double avgPSNR = 0;
    cv::Size outImageSize(0,0);

    for (unsigned long int i = 0; i < inFrames.size()/100; i++ ) 
    {
      // Downscale the output to the original image size, 
      // and do PSNR against original image and downscaled output image.
      cv::resize(outFrames[i],outFrames[i],outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_NN);
      avgPSNR += getPSNR(inFrames[i],outFrames[i]);
    }
    avgPSNR /= inFrames.size()/100;

    std::cout << "Average PSNR per frame: " << avgPSNR << std::endl;
  }

private:

  double scaleFactor;
  std::vector<cv::Mat> inFrames;
  std::vector<cv::Mat> orgFrames;
  std::vector<cv::Mat> outFrames;
  cv::Mat ALD;
  cv::VideoCapture capture;

  cv::Mat sharpen(cv:: Mat input)
  {
      //gradient_bicubic = getGradient(YChannel.clone());
      cv::Mat blurred; double sigma = 1, s_threshold = 5, amount = 1;
      GaussianBlur(input, blurred, cv::Size(), sigma, sigma);
      cv::Mat lowContrastMask = abs(input - blurred) < s_threshold;
      cv::Mat sharpened = input*(1+amount) + blurred*(-amount);
      input.copyTo(sharpened, lowContrastMask);

      return sharpened;
  }

  cv::Mat getDifference(cv::Mat original, cv::Mat HREstimate, cv::Size blurKernel)
  {
    //cv::Mat result;
    cv::Mat HREDownsampled;
    cv::blur(HREstimate, HREDownsampled, blurKernel);
    cv::resize(HREDownsampled,HREDownsampled, cv::Point(0,0), 1/scaleFactor, 1/scaleFactor, CV_INTER_CUBIC);
    //cv::imshow("extracted_edges_blurred",HREDownsampled);
    
    //cv::Mat originalDiff = originalYChannel - HREDownsampled;
    cv::Mat originalDiff = original - HREDownsampled;

    // Upscale and reblur.
    cv::resize(originalDiff,originalDiff, cv::Point(0,0), scaleFactor, scaleFactor, CV_INTER_CUBIC);
    cv::blur(originalDiff, originalDiff, blurKernel);
    //cv::imshow("Difference with original",originalDiff);
    
    return originalDiff;
  } 

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

  float calculateALD(cv::Mat inImage, cv::Point gc, float radius)
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
    if ( (p.x - 1 >= 0) &&
	 (p.y - 1 >= 0) &&
	 (p.x + 1 < img.cols) &&
	 (p.y + 1 < img.rows) )
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
    cv::Mat grad = img.clone();

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

  cv::Mat getALD(cv::Mat imgY, int radius)
  {
    //cv::Mat ALD = imgY;
    cv::Mat ALD(imgY);

    for (int y = 0; y < ALD.rows; y++)
    {
      for (int x = 0; x < ALD.cols; x++)
      {
        ALD.at<unsigned char>(y,x) = (int)calculateALD(ALD, cv::Point(x,y), radius);
      }
    }
    return ALD;
  }

  void sharpenEdges(int i)
  {
    // grab original Y channel.
    cv::Mat YUVIn = inFrames[i];
    std::vector<cv::Mat> originalYUVChannels = convertToYUV(YUVIn);
    cv::Mat originalYChannel = originalYUVChannels[Y];

    // grab bicubic resized Y channel.
    cv::Mat YUVOut = outFrames[i];
    cv::imshow("OriginalRGB",YUVOut);
    std::vector<cv::Mat> YUVChannels = convertToYUV(YUVOut);
   
    cv::imshow("YUVOut",YUVOut);

    cv::Mat YChannel = YUVChannels[Y];
    cv::imshow("Ychannel", YChannel);
    // Use YChannel for rest of algorthm. 

    // Calculate ALD.
    cv::Mat YChannelCopy = YChannel.clone();
   // cv::Mat ALD = getALD(YChannelCopy,15);
    ALD = getALD(YChannelCopy,15);
    cv::imshow("ALD",ALD);
    //YChannel = ALD;
    
    cv::Mat gradient = getGradient(ALD.clone());
    imshow("Gradient",gradient);



    // Extract edges from ALD.
    cv::Mat extractMask;
    cv::Mat extractedEdges;
    cv::adaptiveThreshold(ALD, extractMask, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 5, 1);
    cv::bitwise_not( extractMask, extractMask );
    cv::imshow("extractMask",extractMask);
    //YChannel.copyTo(extractedEdges,extractMask);
    gradient.copyTo(extractedEdges,extractMask);
    cv::imshow("extractedEdges",extractedEdges);

    // Resize image and blur it.
    cv::Size imageSize(0,0);
    cv::resize(extractedEdges,extractedEdges,imageSize,scaleFactor,scaleFactor,CV_INTER_NN);
    cv::blur(extractedEdges, extractedEdges, cv::Size(3,3));
    cv::imshow("extractedEdgesx2",extractedEdges);

    // Apply erosion operator.
    cv::Mat erodedExtractedEdges;
    cv::erode(extractedEdges,erodedExtractedEdges,cv::Mat());
    cv::imshow("erodedExtractedEdges",erodedExtractedEdges);

    // Reblur and downsample. 
    cv::blur(erodedExtractedEdges, erodedExtractedEdges, cv::Size(3,3));
    cv::imshow("erodedEE_sharp", erodedExtractedEdges);
	   
    cv::resize(erodedExtractedEdges,erodedExtractedEdges,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_NN);
    cv::imshow("erodedEE_sharp_downsized", erodedExtractedEdges);

    erodedExtractedEdges *= 4;
    cv::imshow("erodedEE_sharp_downsized * 2", erodedExtractedEdges);
    
      

    cv::Mat downsampledBlurred;
    cv::blur(erodedExtractedEdges, downsampledBlurred, cv::Size(3,3));
    cv::resize(downsampledBlurred,downsampledBlurred,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_NN);
    
    cv::Mat originalDiff = originalYChannel - downsampledBlurred;

    // Upscale and reblur.
    cv::resize(originalDiff,originalDiff,imageSize,scaleFactor,scaleFactor,CV_INTER_NN);
    cv::blur(originalDiff, originalDiff, cv::Size(3,3));
    //originalDiff *= 0.2;

    pow(gradient,2,gradient);
    pow(erodedExtractedEdges,2,erodedExtractedEdges);
    imshow("lol",erodedExtractedEdges - gradient);


std::vector<cv::Mat> orgY = convertToYUV(orgFrames[i]);
    cv::Mat orgYChannel = orgY[Y].clone();
    
     
    //cv::Mat textureArea = finalTexture (orgYChannel, YChannel, scaleFactor);
    



    // Iterative formula
    cv::Mat finalEdgeResult = YChannel + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient);
    //cv::Mat finalEdgeResult = textureArea + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient);
    //cv::Mat finalEdgeResult = originalDiff + YChannel;
    cv::imshow("final",finalEdgeResult);
    //cv::imshow("ychane",YChannel);

//    std::vector<cv::Mat> orgY = convertToYUV(orgFrames[i]);
//    cv::Mat orgYChannel = orgY[Y].clone();
    
     
//    cv::Mat textureArea = finalTexture (orgYChannel, YChannel, scaleFactor);
    

    //Assign modified Y channel back to vector
    YUVChannels[Y] = YChannel;  
    //YUVChannels[Y] = finalEdgeResult + textureArea;  

    // convert back to RGB format.
    cv::Mat RGB = convertToRGB(YUVChannels);
    cv::imshow("RGB",RGB);
    //outFrames[i] = RGB;

    //return finalEdgeResult;
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
  //sv.readVideo(argv[1]);
//  sv.interpolate();
  //sv.resize(argv[3]);
  //sv.writeVideo(argv[2]);
  //sv.testPSNR();
  sv.testSharpenEdges2(argv[3]);


  //Wait until any key is pressed
  cv::waitKey(0);
  std::cout << "Exiting application\n";

  return 0;
}

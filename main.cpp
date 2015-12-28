#include </usr/include/opencv2/opencv.hpp>
#include <bitset>
#include "textureSharpen.cpp"

#define Y 0
#define SHOW_IMAGES 1
//const char* input = "YChannel.png";
const char* input = "me.jpg";
const int ALDRadius = 1;

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

  void resize(char* scaleFactorString, char* bicubic)
  {
    scaleFactor = std::atof(scaleFactorString);
    int resizeBicubic = std::atof(bicubic);
    cv::Size outImageSize(0,0);

    std::cout << "Total # Frames: " << inFrames.size() << std::endl;

    for (unsigned long int i = 0; i < inFrames.size()/50; i++)
    {
      std::cout << "\r" << "Scaling Frame: " << i+1 << "/" << inFrames.size();
      outFrames.push_back(inFrames[i]);
      cv::resize(outFrames[i],outFrames[i],outImageSize,scaleFactor,scaleFactor,CV_INTER_CUBIC);
      //std::cout << i << " " << inFrames[i].rows << " ";;
      //sharpenEdges(i);
     
      // Run a standard resize using bicubic.
      if (resizeBicubic)
      {
        cv::resize(outFrames[i],outFrames[i],outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
      }
      // Use custom implementation.
      else
      {
	sharpenEdgesOrg(i);
      }
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

//
//    // Calculate ALD.
//    cv::Mat YChannelCopy = YChannel.clone();
//    cv::Mat ALD = getALD(YChannelCopy,ALDRadius);
//#if SHOW_IMAGES
//    cv::imshow("ALD",ALD);
//#endif
//    //YChannel = ALD;
//    
//
//    cv::Mat gradient_bicubic = getGradient(ALD.clone());
//#if SHOW_IMAGES
//    cv::imshow("Gradient",gradient_bicubic);
//#endif
//    // G_b = gradient_bicubic.
//
//
//
//    cv::Size blurKernel = cv::Size(3,3);
//
//    // Extract edges from ALD.
//    cv::Mat extractMask = ALD.clone();
//    cv::Mat extractedEdges;
//    //
//    cv::Scalar threshold = cv::mean(ALD);
//    float avgALD = 1.0 * threshold.val[0];
//    //cv::threshold(ALD,extractMask,avgALD, 0, CV_THRESH_TOZERO); 
//
//    cv::Mat gradient_b_dilated;
//    cv::dilate(gradient_bicubic,gradient_b_dilated,cv::Mat());
//    cv::imshow("gbd",gradient_b_dilated);
//
//   for (int y = 0; y < gradient_bicubic.rows; y++) 
//    {
//      for (int x = 0; x < gradient_bicubic.cols; x++) 
//      {
//	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
//	if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  10)
//	{
//          extractMask.at<unsigned char>(y,x) = 255;
//	}
//	else 
//	{
//          extractMask.at<unsigned char>(y,x) = 0;
//	}
//      }
//    }
//    //for (int y = 0; y < ALD.rows; y++) 
//    //{
//    //  for (int x = 0; x < ALD.cols; x++) 
//    //  {
//    //    //grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
//    //    if ((int)ALD.at<unsigned char>(y,x) >  .8*avgALD && (int)ALD.at<unsigned char>(y,x) < 1.2*avgALD)
//    //    {
//    //      extractMask.at<unsigned char>(y,x) = 255;
//    //    }
//    //    else 
//    //    {
//    //      extractMask.at<unsigned char>(y,x) = 0;
//    //    }
//    //  }
//    //}	    
//    //cv::adaptiveThreshold(ALD, extractMask, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 4);
//    //cv::bitwise_not( extractMask, extractMask );
//#if SHOW_IMAGES
//    cv::imshow("extractMask",extractMask);
//#endif
//    //YChannel.copyTo(extractedEdges,extractMask);
//    //gradient_bicubic.copyTo(extractedEdges,extractMask);   //////////////// is this the proper way to mask?
// // cv::multiply(extractedEdges,extractMask,extractedEdges);
//
//
//// standin code - sharpens bicubic gradient map.
//      gradient_bicubic = getGradient(YChannel.clone());
//      cv::Mat blurred; double sigma = 1, s_threshold = 5, amount = 1;
//      GaussianBlur(gradient_bicubic, blurred, cv::Size(), sigma, sigma);
//      cv::Mat lowContrastMask = abs(gradient_bicubic - blurred) < s_threshold;
//      cv::Mat sharpened = gradient_bicubic*(1+amount) + blurred*(-amount);
//      gradient_bicubic.copyTo(sharpened, lowContrastMask);
//  cv::imshow("sdf",sharpened);
//
//    extractedEdges = getGradient(YChannel);
//   // gradient_bicubic.copyTo(extractedEdges,extractMask);   //////////////// is this the proper way to mask?
//
//    //extractedEdges=gradient_bicubic.clone();
//#if HOW_IMAGES
//    cv::imshow("extractedEdges",extractedEdges);
//#endif
//
//    // Resize image and blur it.
//    cv::Size imageSize(0,0);
//    cv::resize(extractedEdges,extractedEdges,imageSize,scaleFactor,scaleFactor,CV_INTER_NN);
//    //cv::blur(extractedEdges, extractedEdges, blurKernel);
//    cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
//    //extractedEdges = sharpen(extractedEdges);
//   // extractedEdges = sharpen(extractedEdges);
//#if SHOW_IMAGES
//    cv::imshow("extractedEdgesx2",extractedEdges);
//#endif
//
//    // Apply erosion operator.
//    cv::Mat erodedExtractedEdges;
//    cv::erode(extractedEdges,erodedExtractedEdges,cv::Mat(),cv::Point(-1,-1),1);
//#if SHOW_IMAGES
//    cv::imshow("erodedExtractedEdges",erodedExtractedEdges);
//#endif
//
//    // Reblur and downsample. 
//    //cv::blur(erodedExtractedEdges, erodedExtractedEdges, blurKernel);
//    cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
//    //erodedExtractedEdges = sharpen(erodedExtractedEdges);
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp", erodedExtractedEdges);
//#endif
//	   
//    cv::resize(erodedExtractedEdges,erodedExtractedEdges,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_CUBIC);
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp_downsized", erodedExtractedEdges);
//#endif
//
//   // erodedExtractedEdges *= 2;  
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp_downsized * 2", erodedExtractedEdges);
//#endif
//    // G_hat = erodedExtractedEdges.
//
//
//
////    cv::Mat HREDownsampled;
////    cv::blur(YChannel, HREDownsampled, blurKernel);
////    cv::resize(HREDownsampled,HREDownsampled,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_NN);
////    cv::imshow("extracted_edges_blurred",HREDownsampled);
////    
////    //cv::Mat originalDiff = originalYChannel - HREDownsampled;
////    cv::Mat originalDiff = originalYChannel - HREDownsampled;
////
////    // Upscale and reblur.
////    cv::resize(originalDiff,originalDiff,imageSize,scaleFactor,scaleFactor,CV_INTER_NN);
////    cv::blur(originalDiff, originalDiff, blurKernel);
////    cv::imshow("Difference with original",originalDiff);
////    //originalDiff *= 0.2;
//
//
//  // standin code - sharpens bicubic gradient map.
//      //gradient_bicubic = getGradient(YChannel.clone());
//      //cv::Mat blurred; double sigma = 1, s_threshold = 5, amount = 1;
//      //GaussianBlur(gradient_bicubic, blurred, cv::Size(), sigma, sigma);
//      //cv::Mat lowContrastMask = abs(gradient_bicubic - blurred) < s_threshold;
//      //cv::Mat sharpened = gradient_bicubic*(1+amount) + blurred*(-amount);
//      //gradient_bicubic.copyTo(sharpened, lowContrastMask);
//#if SHOW_IMAGES
//      //cv::imshow("gb sharp", sharpened);
//      //cv::imshow("low contrast", lowContrastMask);
//#endif
//      //gradient_bicubic = sharpened;
//
//    //pow(gradient_bicubic,2,gradient_bicubic);
//    pow(erodedExtractedEdges,2,erodedExtractedEdges);
//#if SHOW_IMAGES
//    //imshow("Gradient - GradientMap",erodedExtractedEdges - gradient_bicubic);
//#endif
//
//
//   // std::vector<cv::Mat> orgY = convertToYUV(inputImage);
//   // cv::Mat orgYChannel = orgY[Y].clone();
//    
//     
//    //cv::Mat textureArea = finalTexture (orgYChannel, YChannel, scaleFactor);
//
//
//    cv::Mat maskedY; 
//    cv::Mat downSampledExtractMask;
//    cv::resize(extractMask,downSampledExtractMask,cv::Size(0,0),1/scaleFactor,1/scaleFactor,CV_INTER_CUBIC);
//    //multiply(originalYChannel,downSampledExtractMask,maskedY);
//    //multiply(YChannel,extractMask,YChannel);
//    gradient_bicubic = getGradient(YChannel);
//    cv::Mat originalDiff;  
//    // Iterative formula
//    cv::Mat finalEdgeResult;
//    int i = 0;
//    while (i < 50)
//    {
//      //cv::Mat originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
//      originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
//      //originalDiff = getDifference(maskedY, YChannel, blurKernel);
//      //cv::Mat orgYresize;
//      //cv::resize(originalYChannel,orgYresize,cv::Size(0,0),scaleFactor,scaleFactor);
//
//      pow(gradient_bicubic,2,gradient_bicubic);
//
//      //YChannel = orgYresize;
//      YChannel = YChannel + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient_bicubic);
//      //YChannel = YChannel + 0.2*originalDiff + 0.004*(sharpened - gradient_bicubic);
//      
//      //finalEdgeResult = YChannel + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient_bicubic);
//      //cv::Mat finalEdgeResult = textureArea + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient_bicubic);
//      //cv::Mat finalEdgeResult = originalDiff + YChannel;
//      
//      gradient_bicubic = getGradient(YChannel);
//     //cv::GaussianBlur(gradient_bicubic,gradient_bicubic, cv::Size(3, 3), 3);
//      //cv::addWeighted(gradient_bicubic, 1.5, gradient_bicubic, -0.5, 0, gradient_bicubic);
//      //cv::imshow("gradient_last_iteration_sharp",gradient_bicubic);
//
////      cv::pow(gradient_bicubic,2,gradient_bicubic);
////      cv::pow(sharpened,2,sharpened);
////      cv::Mat gradientSubtractionResult;
////
////      cv::subtract(gradient_bicubic,sharpened,gradientSubtractionResult);
////      //cv::absdiff(gradient_bicubic,sharpened,gradientSubtractionResult);
////      cv::imshow("Gradient - GradientMap 2", gradientSubtractionResult);
//      i++;
//    }
//    //cv::imshow("final",finalEdgeResult);
//#if SHOW_IMAGES
//    cv::imshow("YchannelFinal",YChannel);
//      cv::imshow("gradient_last_iteration",gradient_bicubic);
//      cv::imshow("originalDiff",originalDiff);
//#endif
// 
////    std::vector<cv::Mat> orgY = convertToYUV(orgFrames[i]);
////    cv::Mat orgYChannel = orgY[Y].clone();
//    
//     
////    cv::Mat textureArea = finalTexture (orgYChannel, YChannel, scaleFactor);
//    
//
    //Assign modified Y channel back to vector
    YUVChannels[Y] = YChannel;  

    // convert back to RGB format.
    cv::Mat processedRGB = convertToRGB(YUVChannels);
#if SHOW_IMAGES
    cv::imshow("processedRGB",processedRGB);
#endif

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
    
    // downscale original so we can test the uprezzed.
    cv::Mat inputDownscaled;
    cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    //cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);

    // upscale bicubic downscaled image using bicubic.
    cv::Mat inputBICUBIC;
    cv::resize(inputDownscaled,inputBICUBIC,outImageSize,scaleFactor,scaleFactor,CV_INTER_CUBIC);

    cv::Mat YUVIn = inputDownscaled.clone();
    std::vector<cv::Mat> originalYUVChannels = convertToYUV(YUVIn);
    cv::Mat originalYChannel = originalYUVChannels[Y];
    // input bicubic is real resized bicubic?????????????????

    // Try straight conversion to YUV->RGB and compare PSNR.
    // Turns out PSNR decreases slightly when converting from RGB->YUV-RGB. Due to opencv bug.
    cv::Mat BY = inputBICUBIC.clone();
    std::vector<cv::Mat> BYY = convertToYUV(BY);
    cv::Mat RGBOrg = convertToRGB(BYY);


   
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
    

    cv::Mat original_gradient;
    cv::Mat gradient_bicubic;
    gradient_bicubic = getGradient(ALD.clone());
#if SHOW_IMAGES
    cv::imshow("Gradient ALD",gradient_bicubic);
#endif
    // G_b = gradient_bicubic.
    gradient_bicubic = getGradient(YChannel.clone());
    original_gradient = getGradient(BYY[Y]);
#if SHOW_IMAGES
    cv::imshow("Gradient bicubic",gradient_bicubic);
    cv::imshow("Gradient original",original_gradient);
#endif
 

    cv::Size blurKernel = cv::Size(3,3);

    // Extract edges from ALD.
    cv::Mat extractMask = ALD.clone();
    cv::Mat extractedEdges;
    extractedEdges = getGradient(YChannel);
    
    cv::Scalar threshold = cv::mean(ALD);
    float avgALD = 1.5 * threshold.val[0];

    cv::Mat gradient_b_dilated;
    cv::dilate(gradient_bicubic,gradient_b_dilated,cv::Mat());
    cv::imshow("gbd",gradient_b_dilated);



    cv::Mat extractALD = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    for (int y = 0; y < gradient_bicubic.rows; y++) 
    {
      for (int x = 0; x < gradient_bicubic.cols; x++) 
      {
	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
	if ((int)ALD.at<unsigned char>(y,x) >  avgALD)
	{
          extractALD.at<unsigned char>(y,x) = 255;
	}
	else 
	{
          extractALD.at<unsigned char>(y,x) = 0;
	}
      }
    }
    cv::imshow("extractALD",extractALD);



    for (int y = 0; y < gradient_bicubic.rows; y++) 
    {
      for (int x = 0; x < gradient_bicubic.cols; x++) 
      {
	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
	if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  20)
	{
          extractMask.at<unsigned char>(y,x) = 255;
	}
	else 
	{
          extractMask.at<unsigned char>(y,x) = 0;
	}
      }
    }

#if SHOW_IMAGES
    cv::imshow("extractMask",extractMask);
#endif
    for (int y = 0; y < gradient_bicubic.rows; y++) 
    {
      for (int x = 0; x < gradient_bicubic.cols; x++) 
      {
	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
	if ((int)extractMask.at<unsigned char>(y,x) <  10)
	{
          //gradient_bicubic.at<unsigned char>(y,x) = 0;
          extractedEdges.at<unsigned char>(y,x) = 0;
        }
      }
    }

       

#if HOW_IMAGES
    cv::imshow("extractedEdges",extractedEdges);
#endif

    // Resize image and blur it.
    cv::Size imageSize(0,0);
    cv::resize(extractedEdges,extractedEdges,imageSize,scaleFactor,scaleFactor,CV_INTER_NN);
    //cv::blur(extractedEdges, extractedEdges, blurKernel);
    //cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
    extractedEdges = sharpen(extractedEdges);
#if SHOW_IMAGES
    cv::imshow("extractedEdges_upscaled",extractedEdges);
#endif

    // Apply erosion operator.
    cv::Mat erodedExtractedEdges;
    cv::erode(extractedEdges,erodedExtractedEdges,cv::Mat(),cv::Point(-1,-1),1);
#if SHOW_IMAGES
    cv::imshow("erodedExtractedEdges",erodedExtractedEdges);
#endif
    // Reblur and downsample. 
    //cv::blur(erodedExtractedEdges, erodedExtractedEdges, blurKernel);
    //cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
    //cv::GaussianBlur(erodedExtractedEdges,erodedExtractedEdges,cv::Size(11,11),1,1);
    //erodedExtractedEdges = sharpen(erodedExtractedEdges);
    erodedExtractedEdges *= 5000.0;  
#if SHOW_IMAGES
    cv::imshow("erodedEE_sharp", erodedExtractedEdges);
#endif
	   
    cv::resize(erodedExtractedEdges,erodedExtractedEdges,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_CUBIC);
#if SHOW_IMAGES
    cv::imshow("erodedEE_sharp_downscaled", erodedExtractedEdges);
#endif
   
    erodedExtractedEdges *= 1.0;  
#if SHOW_IMAGES
    cv::imshow("erodedEE_sharp_downscaled * 2", erodedExtractedEdges);
#endif
    // G_hat = erodedExtractedEdges.

#if SHOW_IMAGES
      //cv::imshow("gb sharp", sharpened);
      //cv::imshow("low contrast", lowContrastMask);
#endif
      //gradient_bicubic = sharpened;


    cv::Mat showb = gradient_bicubic.clone();
    cv::Mat showe = erodedExtractedEdges.clone();
    cv::Mat showMinus = erodedExtractedEdges.clone();

    cv::pow(showe,2,showe);
    cv::pow(showb,2,showb);

    //showMinus = (erodedExtractedEdges - showb);
    //cv::subtract(showb, erodedExtractedEdges,showMinus);
    
  //  for (int y = 0; y < gradient_bicubic.rows; y++) 
  //  {
  //    for (int x = 0; x < gradient_bicubic.cols; x++) 
  //    {
  //      //grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
  //      showMinus.at<unsigned char>(y,x) = 100;
  //      //erodedExtractedEdges.at<unsigned char>(y,x) = 0;
  //      //erodedExtractedEdges.at<unsigned char>(y,x) - 
  //      //gradient_bicubic.at<unsigned char>(y,x);
  //      //std::cout << ".";	
  //    }
  //  }
    
    cv::absdiff(showe,showb,showMinus);
    //showMinus*=50;
#if SHOW_IMAGES
    cv::imshow("Gradient - GradientMap",showe - showb);
    cv::imshow("Gradient - GradientMap2",showMinus);
    cv::imshow("Gradient extract^2",showe);
    cv::imshow("Gradient org^2",showb);
    //imshow("Gradient*Y",showMinus*YChannel.clone());
#endif

    cv::Mat originalDiff;  
    cv::Mat graDiff;  

    cv::pow(erodedExtractedEdges,2,erodedExtractedEdges);
    
    // Iterative formula
    int i = 0;
    while (i < 50)
    {
      //originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
     // pow(erodedExtractedEdges,2,erodedExtractedEdges);
      pow(gradient_bicubic,2,gradient_bicubic);
      cv::absdiff(erodedExtractedEdges,gradient_bicubic,graDiff);
      cv::imshow("gb-gh", gradient_bicubic - erodedExtractedEdges);
      cv::imshow("gh-gb", erodedExtractedEdges - gradient_bicubic);

      //YChannel = YChannel;
      //YChannel = YChannel + 0.2*originalDiff + 0.004*(graDiff);
      //YChannel = YChannel + 0.2*originalDiff + 0.004*(graDiff);
      //YChannel = YChannel + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient_bicubic);
      //YChannel = YChannel + 40.4*(erodedExtractedEdges - gradient_bicubic);
      //YChannel = YChannel + 0.20*originalDiff  + 0.004*(erodedExtractedEdges - gradient_bicubic);
      //YChannel = YChannel + 900*YChannel.mul(erodedExtractedEdges - gradient_bicubic,1);
      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1) - 0.002*originalDiff;

      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1);
      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1);

      //YChannel = YChannel - 0.4*originalDiff;
      //
      //YChannel = YChannel + YChannel.absdiff(originalDiff,1);
      //YChannel = YChannel + YChannel.mul(graDiff,1);


      //YChannel = YChannel + 0.2*originalDiff;

      //YChannel = YChannel + (erodedExtractedEdges - gradient_bicubic);
      
      // gradient compensated sharp edges.
      //YChannel = YChannel + 400*(erodedExtractedEdges - gradient_bicubic);

      
      // NOTE: get difference currently modifies the YChannel result!
      originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
      gradient_bicubic = getGradient(YChannel);
      i++;
    }

#if SHOW_IMAGES
    cv::imshow("erod-grad",(erodedExtractedEdges - gradient_bicubic));
    cv::imshow("grad-erod",(gradient_bicubic - erodedExtractedEdges));
    cv::imshow("YchannelFinal",YChannel);
    cv::imshow("graDiff",graDiff);
    cv::imshow("gradient_last_iteration",gradient_bicubic);
    cv::imshow("originalDiff",originalDiff);
    cv::imshow("grad*Y",YChannel.mul(graDiff,0.5));
    //cv::subtract(YChannel,originalDiff,YChannel);
    //cv::imshow("sum Y - OD",YChannel);
#endif
    
    
    

    //cv::GaussianBlur(YChannel,YChannel,cv::Size(11,11),1,1); 
    //cv::imshow("YChannel1d blur", YChannel);
    //Assign modified Y channel back to vector
    //YUVChannels[Y] = YChannel;  
    YUVChannels[Y] = YChannel; 
    
   getFinalTexture(originalYChannel,YChannel,ALD,scaleFactor);
    // convert back to RGB format.
    cv::Mat processedRGB = convertToRGB(YUVChannels);
#if SHOW_IMAGES
    cv::imshow("processedRGB",processedRGB);
#endif

   // std::cout << "bicubic PSNR: " << getPSNR(inputImage,inputBICUBIC) << "\n";
    std::cout << "bicubic PSNR: " << getPSNR(inputImage,RGBOrg) << "\n";
    std::cout << "sharpened PSNR: " << getPSNR(inputImage,processedRGB) << "\n";
  }

  void testSharpenEdges3(char* scaleFactorString)
  {
    scaleFactor = std::atof(scaleFactorString);
    cv::Size outImageSize(0,0);

    // grab original Y channel.
    cv::Mat inputImage = cv::imread(input);
    if (inputImage.rows == 0) std::cout << "NOPE\n";
    
    // downscale original so we can test the uprezzed.
    cv::Mat inputDownscaled;
    cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    //cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);

    // upscale bicubic downscaled image using bicubic.
    cv::Mat inputBICUBIC;
    cv::resize(inputDownscaled,inputBICUBIC,outImageSize,scaleFactor,scaleFactor,CV_INTER_CUBIC);

    cv::Mat YUVIn = inputDownscaled.clone();
    std::vector<cv::Mat> originalYUVChannels = convertToYUV(YUVIn);
    cv::Mat originalYChannel = originalYUVChannels[Y];
    // input bicubic is real resized bicubic?????????????????

    // Try straight conversion to YUV->RGB and compare PSNR.
    // Turns out PSNR decreases slightly when converting from RGB->YUV-RGB. Due to opencv bug.
    cv::Mat BY = inputBICUBIC.clone();
    std::vector<cv::Mat> BYY = convertToYUV(BY);
    cv::Mat RGBOrg = convertToRGB(BYY);


   
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
    

    cv::Mat original_gradient;
    cv::Mat gradient_bicubic;
    gradient_bicubic = getGradient(ALD.clone());
#if SHOW_IMAGES
    cv::imshow("Gradient ALD",gradient_bicubic);
#endif
    // G_b = gradient_bicubic.
    gradient_bicubic = getGradient(YChannel.clone());
    original_gradient = getGradient(BYY[Y]);
#if SHOW_IMAGES
    cv::imshow("Gradient bicubic",gradient_bicubic);
    cv::imshow("Gradient original",original_gradient);
#endif
 

    cv::Size blurKernel = cv::Size(3,3);

    // Extract edges from ALD.
    cv::Mat extractMask = ALD.clone();
    cv::Mat extractedEdges;
    extractedEdges = getGradient(YChannel);
    
    cv::Scalar threshold = cv::mean(ALD);
    float avgALD = 1.5 * threshold.val[0];

    cv::Mat gradient_b_dilated;
    cv::dilate(gradient_bicubic,gradient_b_dilated,cv::Mat());
    cv::imshow("gbd",gradient_b_dilated);


    //for (int y = 0; y < gradient_bicubic.rows; y++) 
    //{
    //  for (int x = 0; x < gradient_bicubic.cols; x++) 
    //  {
    //    //grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
    //    if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  20)
    //    {
    //      extractMask.at<unsigned char>(y,x) = 255;
    //    }
    //    else 
    //    {
    //      extractMask.at<unsigned char>(y,x) = 0;
    //    }
    //  }
    //}
    //imshow("goodGradMap", extractMask);



    cv::Mat maskedY = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat extractALD = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    for (int y = 0; y < gradient_bicubic.rows; y++) 
    {
      for (int x = 0; x < gradient_bicubic.cols; x++) 
      {
	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
	if ((int)ALD.at<unsigned char>(y,x) >  avgALD)
	{
          //extractALD.at<unsigned char>(y,x) = 255;
	  maskedY.at<unsigned char>(y,x) = YChannel.at<unsigned char>(y,x);
	}
	else 
	{
          extractALD.at<unsigned char>(y,x) = 0;
	}
      }
    }


    cv::imshow("extractALD",extractALD);
    //cv::multiply(YChannel.clone(),extractALD,maskedY,1);
    //YChannel.copyTo(maskedY,extractALD);
    cv::imshow("maskedY",maskedY);
    
    cv::Mat gradMaskY = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    gradMaskY = getGradient(maskedY);
    cv::imshow("gradMaskY",gradMaskY);

    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
    cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),0,1);
    cv::Mat erodedMaskY;
    cv::imshow("gradMaskpreero",gradMaskY);
    cv::erode(gradMaskY,erodedMaskY,cv::Mat(),cv::Point(-1,-1),1);
    cv::imshow("erodedMaskedY",erodedMaskY);
    cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),1,0);
    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    gradMaskY *= 2;
    cv::imshow("erodedMaskedFinal",gradMaskY);

    

    cv::imshow("YchanelnoBlur",YChannel);
    cv::Mat testblur = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::GaussianBlur(YChannel,testblur,cv::Size(25,25),0,9);
    cv::imshow("testblurY",testblur);
    cv::GaussianBlur(YChannel,testblur,cv::Size(25,25),9,0);
    cv::imshow("testblurX",testblur);

    //cv::Mat gradEroded  = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    //gradEroded = getGradient(YChannel);

    cv::Mat Y2;
    cv::Mat Xt = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat XtD = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat diff = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gradB = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gb2 = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gh2 = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    Y2 = originalYChannel.clone(); 
    Xt = YChannel.clone(); 
    XtD = YChannel.clone(); 
    gradB = getGradient(Xt);
     



    int iterator = 0;
    while (iterator < 50)
    {
      cv::GaussianBlur(Xt,XtD,cv::Size(5,5),1,0);
      cv::resize(XtD,XtD,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
      diff = Y2 - XtD;
      cv::resize(diff,diff,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
      cv::GaussianBlur(diff,diff,cv::Size(5,5),0,1);

      gradB = getGradient(Xt);
      cv::pow(gradB,2,gb2);
      cv::pow(gradMaskY,2,gh2);

      Xt = Xt + 0.2*diff + 0.004*(gradMaskY - gradB); 
      iterator++;
    }

   imshow("Xt",Xt);
  // YUVChannels[Y] = Xt; 
   //YUVChannels[Y] = YChannel; 
    
   //getFinalTexture(originalYChannel,YChannel,ALD,scaleFactor);
   cv::Mat finalImage = getFinalTexture(originalYChannel,Xt,ALD,scaleFactor);
   YUVChannels[Y] = finalImage; 

    // convert back to RGB format.
   cv::Mat processedRGB = convertToRGB(YUVChannels);
   std::cout << "Orgbicubic PSNR: " << getPSNR(inputImage,RGBOrg) << "\n";
   std::cout << "Orgsharpened PSNR: " << getPSNR(inputImage,processedRGB) << "\n";
  
//
//    for (int y = 0; y < gradient_bicubic.rows; y++) 
//    {
//      for (int x = 0; x < gradient_bicubic.cols; x++) 
//      {
//	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
//	if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  20)
//	{
//          extractMask.at<unsigned char>(y,x) = 255;
//	}
//	else 
//	{
//          extractMask.at<unsigned char>(y,x) = 0;
//	}
//      }
//    }
//
//#if SHOW_IMAGES
//    cv::imshow("extractMask",extractMask);
//#endif
//    for (int y = 0; y < gradient_bicubic.rows; y++) 
//    {
//      for (int x = 0; x < gradient_bicubic.cols; x++) 
//      {
//	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
//	if ((int)extractMask.at<unsigned char>(y,x) <  10)
//	{
//          //gradient_bicubic.at<unsigned char>(y,x) = 0;
//          extractedEdges.at<unsigned char>(y,x) = 0;
//        }
//      }
//    }
//
//       
//
//#if SHOW_IMAGES
//    cv::imshow("extractedEdges",extractedEdges);
//#endif
//
//    // Resize image and blur it.
//    cv::Size imageSize(0,0);
//    cv::resize(extractedEdges,extractedEdges,imageSize,scaleFactor,scaleFactor,CV_INTER_NN);
//    //cv::blur(extractedEdges, extractedEdges, blurKernel);
//    //cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
//    extractedEdges = sharpen(extractedEdges);
//#if SHOW_IMAGES
//    cv::imshow("extractedEdges_upscaled",extractedEdges);
//#endif
//
//    // Apply erosion operator.
//    cv::Mat erodedExtractedEdges;
//    cv::erode(extractedEdges,erodedExtractedEdges,cv::Mat(),cv::Point(-1,-1),1);
//#if SHOW_IMAGES
//    cv::imshow("erodedExtractedEdges",erodedExtractedEdges);
//#endif
//    // Reblur and downsample. 
//    //cv::blur(erodedExtractedEdges, erodedExtractedEdges, blurKernel);
//    //cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
//    //cv::GaussianBlur(erodedExtractedEdges,erodedExtractedEdges,cv::Size(11,11),1,1);
//    //erodedExtractedEdges = sharpen(erodedExtractedEdges);
//    erodedExtractedEdges *= 5000.0;  
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp", erodedExtractedEdges);
//#endif
//	   
//    cv::resize(erodedExtractedEdges,erodedExtractedEdges,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_CUBIC);
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp_downscaled", erodedExtractedEdges);
//#endif
//   
//    erodedExtractedEdges *= 1.0;  
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp_downscaled * 2", erodedExtractedEdges);
//#endif
//    // G_hat = erodedExtractedEdges.
//
//#if SHOW_IMAGES
//      //cv::imshow("gb sharp", sharpened);
//      //cv::imshow("low contrast", lowContrastMask);
//#endif
//      //gradient_bicubic = sharpened;
//
//
//    cv::Mat showb = gradient_bicubic.clone();
//    cv::Mat showe = erodedExtractedEdges.clone();
//    cv::Mat showMinus = erodedExtractedEdges.clone();
//
//    cv::pow(showe,2,showe);
//    cv::pow(showb,2,showb);
//
//    //showMinus = (erodedExtractedEdges - showb);
//    //cv::subtract(showb, erodedExtractedEdges,showMinus);
//    
//  //  for (int y = 0; y < gradient_bicubic.rows; y++) 
//  //  {
//  //    for (int x = 0; x < gradient_bicubic.cols; x++) 
//  //    {
//  //      //grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
//  //      showMinus.at<unsigned char>(y,x) = 100;
//  //      //erodedExtractedEdges.at<unsigned char>(y,x) = 0;
//  //      //erodedExtractedEdges.at<unsigned char>(y,x) - 
//  //      //gradient_bicubic.at<unsigned char>(y,x);
//  //      //std::cout << ".";	
//  //    }
//  //  }
//    
//    cv::absdiff(showe,showb,showMinus);
//    //showMinus*=50;
//#if SHOW_IMAGES
//    cv::imshow("Gradient - GradientMap",showe - showb);
//    cv::imshow("Gradient - GradientMap2",showMinus);
//    cv::imshow("Gradient extract^2",showe);
//    cv::imshow("Gradient org^2",showb);
//    //imshow("Gradient*Y",showMinus*YChannel.clone());
//#endif
//
//    cv::Mat originalDiff;  
//    cv::Mat graDiff;  
//
//    cv::pow(erodedExtractedEdges,2,erodedExtractedEdges);
//    
//    // Iterative formula
//    int i = 0;
//    while (i < 50)
//    {
//      //originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
//     // pow(erodedExtractedEdges,2,erodedExtractedEdges);
//      pow(gradient_bicubic,2,gradient_bicubic);
//      cv::absdiff(erodedExtractedEdges,gradient_bicubic,graDiff);
//      cv::imshow("gb-gh", gradient_bicubic - erodedExtractedEdges);
//      cv::imshow("gh-gb", erodedExtractedEdges - gradient_bicubic);
//
//      //YChannel = YChannel;
//      //YChannel = YChannel + 0.2*originalDiff + 0.004*(graDiff);
//      //YChannel = YChannel + 0.2*originalDiff + 0.004*(graDiff);
//      //YChannel = YChannel + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient_bicubic);
//      //YChannel = YChannel + 40.4*(erodedExtractedEdges - gradient_bicubic);
//      //YChannel = YChannel + 0.20*originalDiff  + 0.004*(erodedExtractedEdges - gradient_bicubic);
//      //YChannel = YChannel + 900*YChannel.mul(erodedExtractedEdges - gradient_bicubic,1);
//      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1) - 0.002*originalDiff;
//
//      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1);
//      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1);
//
//      //YChannel = YChannel - 0.4*originalDiff;
//      //
//      //YChannel = YChannel + YChannel.absdiff(originalDiff,1);
//      //YChannel = YChannel + YChannel.mul(graDiff,1);
//
//
//      //YChannel = YChannel + 0.2*originalDiff;
//
//      //YChannel = YChannel + (erodedExtractedEdges - gradient_bicubic);
//      
//      // gradient compensated sharp edges.
//      //YChannel = YChannel + 400*(erodedExtractedEdges - gradient_bicubic);
//
//      
//      // NOTE: get difference currently modifies the YChannel result!
//      originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
//      gradient_bicubic = getGradient(YChannel);
//      i++;
//    }
//
//#if SHOW_IMAGES
//    cv::imshow("erod-grad",(erodedExtractedEdges - gradient_bicubic));
//    cv::imshow("grad-erod",(gradient_bicubic - erodedExtractedEdges));
//    cv::imshow("YchannelFinal",YChannel);
//    cv::imshow("graDiff",graDiff);
//    cv::imshow("gradient_last_iteration",gradient_bicubic);
//    cv::imshow("originalDiff",originalDiff);
//    cv::imshow("grad*Y",YChannel.mul(graDiff,0.5));
//    //cv::subtract(YChannel,originalDiff,YChannel);
//    //cv::imshow("sum Y - OD",YChannel);
//#endif
//    
//    
//    
//
//    //cv::GaussianBlur(YChannel,YChannel,cv::Size(11,11),1,1); 
//    //cv::imshow("YChannel1d blur", YChannel);
//    //Assign modified Y channel back to vector
//    //YUVChannels[Y] = YChannel;  
//    YUVChannels[Y] = YChannel; 
//    
//   getFinalTexture(originalYChannel,YChannel,ALD,scaleFactor);
//    // convert back to RGB format.
//    cv::Mat processedRGB = convertToRGB(YUVChannels);
//#if SHOW_IMAGES
//    cv::imshow("processedRGB",processedRGB);
//#endif
//
//   // std::cout << "bicubic PSNR: " << getPSNR(inputImage,inputBICUBIC) << "\n";
//    std::cout << "bicubic PSNR: " << getPSNR(inputImage,RGBOrg) << "\n";
//    std::cout << "sharpened PSNR: " << getPSNR(inputImage,processedRGB) << "\n";
  }

  void testSharpenEdges4(char* scaleFactorString)
  {
    scaleFactor = std::atof(scaleFactorString);
    cv::Size outImageSize(0,0);

    // grab original Y channel.
    cv::Mat inputImage = cv::imread(input);
    if (inputImage.rows == 0) std::cout << "NOPE\n";
    
    // downscale original so we can test the uprezzed.
    cv::Mat inputDownscaled;
    cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    //cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);

    // upscale bicubic downscaled image using bicubic.
    cv::Mat inputBICUBIC;
    cv::resize(inputDownscaled,inputBICUBIC,outImageSize,scaleFactor,scaleFactor,CV_INTER_CUBIC);

    cv::Mat YUVIn = inputDownscaled.clone();
    std::vector<cv::Mat> originalYUVChannels = convertToYUV(YUVIn);
    cv::Mat originalYChannel = originalYUVChannels[Y];
    // input bicubic is real resized bicubic?????????????????

    // Try straight conversion to YUV->RGB and compare PSNR.
    // Turns out PSNR decreases slightly when converting from RGB->YUV-RGB. Due to opencv bug.
    cv::Mat BY = inputBICUBIC.clone();
    std::vector<cv::Mat> BYY = convertToYUV(BY);
    cv::Mat RGBOrg = convertToRGB(BYY);


   
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
    

    cv::Mat original_gradient;
    cv::Mat gradient_bicubic;
    gradient_bicubic = getGradient(ALD.clone());
#if SHOW_IMAGES
    cv::imshow("Gradient ALD",gradient_bicubic);
#endif
    // G_b = gradient_bicubic.
    gradient_bicubic = getGradient(YChannel.clone());
    original_gradient = getGradient(BYY[Y]);
#if SHOW_IMAGES
    cv::imshow("Gradient bicubic",gradient_bicubic);
    cv::imshow("Gradient original",original_gradient);
#endif
 

    cv::Size blurKernel = cv::Size(3,3);

    // Extract edges from ALD.
    cv::Mat extractMask = ALD.clone();
    cv::Mat extractedEdges;
    extractedEdges = getGradient(YChannel);
    
    cv::Scalar threshold = cv::mean(ALD);
    float avgALD = 1.5 * threshold.val[0];

    cv::Mat gradient_b_dilated;
    cv::dilate(gradient_bicubic,gradient_b_dilated,cv::Mat());
    cv::imshow("gbd",gradient_b_dilated);


    //for (int y = 0; y < gradient_bicubic.rows; y++) 
    //{
    //  for (int x = 0; x < gradient_bicubic.cols; x++) 
    //  {
    //    //grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
    //    if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  20)
    //    {
    //      extractMask.at<unsigned char>(y,x) = 255;
    //    }
    //    else 
    //    {
    //      extractMask.at<unsigned char>(y,x) = 0;
    //    }
    //  }
    //}
    //imshow("goodGradMap", extractMask);



    cv::Mat maskedY = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat extractALD = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    for (int y = 0; y < gradient_bicubic.rows; y++) 
    {
      for (int x = 0; x < gradient_bicubic.cols; x++) 
      {
	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
	if ((int)ALD.at<unsigned char>(y,x) >  avgALD)
	{
          //extractALD.at<unsigned char>(y,x) = 255;
	  maskedY.at<unsigned char>(y,x) = YChannel.at<unsigned char>(y,x);
	}
	else 
	{
          extractALD.at<unsigned char>(y,x) = 0;
	}
      }
    }


    cv::imshow("extractALD",extractALD);
    //cv::multiply(YChannel.clone(),extractALD,maskedY,1);
    //YChannel.copyTo(maskedY,extractALD);
    cv::imshow("maskedY",maskedY);
    
    cv::Mat gradMaskY = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    gradMaskY = getGradient(maskedY);
    cv::imshow("gradMaskY",gradMaskY);

    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
    cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),0,1);
    cv::Mat erodedMaskY;
    cv::imshow("gradMaskpreero",gradMaskY);
    cv::erode(gradMaskY,erodedMaskY,cv::Mat(),cv::Point(-1,-1),1);
    cv::imshow("erodedMaskedY",erodedMaskY);
    cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),1,0);
    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    gradMaskY *= 2;
    cv::imshow("erodedMaskedFinal",gradMaskY);

    

    cv::imshow("YchanelnoBlur",YChannel);
    cv::Mat testblur = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::GaussianBlur(YChannel,testblur,cv::Size(25,25),0,9);
    cv::imshow("testblurY",testblur);
    cv::GaussianBlur(YChannel,testblur,cv::Size(25,25),9,0);
    cv::imshow("testblurX",testblur);

    //cv::Mat gradEroded  = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    //gradEroded = getGradient(YChannel);

    cv::Mat Y2;
    cv::Mat Xt = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat XtD = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat diff = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gradB = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gb2 = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gh2 = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    Y2 = originalYChannel.clone(); 
    Xt = YChannel.clone(); 
    XtD = YChannel.clone(); 
    gradB = getGradient(Xt);
     



    int iterator = 0;
    while (iterator < 50)
    {
      cv::GaussianBlur(Xt,XtD,cv::Size(5,5),1,0);
      cv::resize(XtD,XtD,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
      diff = Y2 - XtD;
      cv::resize(diff,diff,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
      cv::GaussianBlur(diff,diff,cv::Size(5,5),0,1);

      gradB = getGradient(Xt);
      cv::pow(gradB,2,gb2);
      cv::pow(gradMaskY,2,gh2);

      Xt = Xt + 0.2*diff + 0.004*(gradMaskY - gradB); 
      iterator++;
    }

   imshow("Xt",Xt);
   YUVChannels[Y] = Xt; 
   //YUVChannels[Y] = YChannel; 
    
   //getFinalTexture(originalYChannel,YChannel,ALD,scaleFactor);
   
   //cv::Mat finalImage = getFinalTexture(originalYChannel,Xt,ALD,scaleFactor);
   ////cv::Mat finalImage = getFinalTexture(originalYChannel,YChannel,ALD,scaleFactor);
  // YUVChannels[Y] = finalImage; 
  // imshow("Final iamge",finalImage);

    // convert back to RGB format.
   cv::Mat processedRGB = convertToRGB(YUVChannels);
   std::cout << "Orgbicubic PSNR: " << getPSNR(inputImage,RGBOrg) << "\n";
   std::cout << "Orgsharpened PSNR: " << getPSNR(inputImage,processedRGB) << "\n";
  
//
//    for (int y = 0; y < gradient_bicubic.rows; y++) 
//    {
//      for (int x = 0; x < gradient_bicubic.cols; x++) 
//      {
//	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
//	if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  20)
//	{
//          extractMask.at<unsigned char>(y,x) = 255;
//	}
//	else 
//	{
//          extractMask.at<unsigned char>(y,x) = 0;
//	}
//      }
//    }
//
//#if SHOW_IMAGES
//    cv::imshow("extractMask",extractMask);
//#endif
//    for (int y = 0; y < gradient_bicubic.rows; y++) 
//    {
//      for (int x = 0; x < gradient_bicubic.cols; x++) 
//      {
//	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
//	if ((int)extractMask.at<unsigned char>(y,x) <  10)
//	{
//          //gradient_bicubic.at<unsigned char>(y,x) = 0;
//          extractedEdges.at<unsigned char>(y,x) = 0;
//        }
//      }
//    }
//
//       
//
//#if SHOW_IMAGES
//    cv::imshow("extractedEdges",extractedEdges);
//#endif
//
//    // Resize image and blur it.
//    cv::Size imageSize(0,0);
//    cv::resize(extractedEdges,extractedEdges,imageSize,scaleFactor,scaleFactor,CV_INTER_NN);
//    //cv::blur(extractedEdges, extractedEdges, blurKernel);
//    //cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
//    extractedEdges = sharpen(extractedEdges);
//#if SHOW_IMAGES
//    cv::imshow("extractedEdges_upscaled",extractedEdges);
//#endif
//
//    // Apply erosion operator.
//    cv::Mat erodedExtractedEdges;
//    cv::erode(extractedEdges,erodedExtractedEdges,cv::Mat(),cv::Point(-1,-1),1);
//#if SHOW_IMAGES
//    cv::imshow("erodedExtractedEdges",erodedExtractedEdges);
//#endif
//    // Reblur and downsample. 
//    //cv::blur(erodedExtractedEdges, erodedExtractedEdges, blurKernel);
//    //cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
//    //cv::GaussianBlur(erodedExtractedEdges,erodedExtractedEdges,cv::Size(11,11),1,1);
//    //erodedExtractedEdges = sharpen(erodedExtractedEdges);
//    erodedExtractedEdges *= 5000.0;  
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp", erodedExtractedEdges);
//#endif
//	   
//    cv::resize(erodedExtractedEdges,erodedExtractedEdges,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_CUBIC);
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp_downscaled", erodedExtractedEdges);
//#endif
//   
//    erodedExtractedEdges *= 1.0;  
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp_downscaled * 2", erodedExtractedEdges);
//#endif
//    // G_hat = erodedExtractedEdges.
//
//#if SHOW_IMAGES
//      //cv::imshow("gb sharp", sharpened);
//      //cv::imshow("low contrast", lowContrastMask);
//#endif
//      //gradient_bicubic = sharpened;
//
//
//    cv::Mat showb = gradient_bicubic.clone();
//    cv::Mat showe = erodedExtractedEdges.clone();
//    cv::Mat showMinus = erodedExtractedEdges.clone();
//
//    cv::pow(showe,2,showe);
//    cv::pow(showb,2,showb);
//
//    //showMinus = (erodedExtractedEdges - showb);
//    //cv::subtract(showb, erodedExtractedEdges,showMinus);
//    
//  //  for (int y = 0; y < gradient_bicubic.rows; y++) 
//  //  {
//  //    for (int x = 0; x < gradient_bicubic.cols; x++) 
//  //    {
//  //      //grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
//  //      showMinus.at<unsigned char>(y,x) = 100;
//  //      //erodedExtractedEdges.at<unsigned char>(y,x) = 0;
//  //      //erodedExtractedEdges.at<unsigned char>(y,x) - 
//  //      //gradient_bicubic.at<unsigned char>(y,x);
//  //      //std::cout << ".";	
//  //    }
//  //  }
//    
//    cv::absdiff(showe,showb,showMinus);
//    //showMinus*=50;
//#if SHOW_IMAGES
//    cv::imshow("Gradient - GradientMap",showe - showb);
//    cv::imshow("Gradient - GradientMap2",showMinus);
//    cv::imshow("Gradient extract^2",showe);
//    cv::imshow("Gradient org^2",showb);
//    //imshow("Gradient*Y",showMinus*YChannel.clone());
//#endif
//
//    cv::Mat originalDiff;  
//    cv::Mat graDiff;  
//
//    cv::pow(erodedExtractedEdges,2,erodedExtractedEdges);
//    
//    // Iterative formula
//    int i = 0;
//    while (i < 50)
//    {
//      //originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
//     // pow(erodedExtractedEdges,2,erodedExtractedEdges);
//      pow(gradient_bicubic,2,gradient_bicubic);
//      cv::absdiff(erodedExtractedEdges,gradient_bicubic,graDiff);
//      cv::imshow("gb-gh", gradient_bicubic - erodedExtractedEdges);
//      cv::imshow("gh-gb", erodedExtractedEdges - gradient_bicubic);
//
//      //YChannel = YChannel;
//      //YChannel = YChannel + 0.2*originalDiff + 0.004*(graDiff);
//      //YChannel = YChannel + 0.2*originalDiff + 0.004*(graDiff);
//      //YChannel = YChannel + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient_bicubic);
//      //YChannel = YChannel + 40.4*(erodedExtractedEdges - gradient_bicubic);
//      //YChannel = YChannel + 0.20*originalDiff  + 0.004*(erodedExtractedEdges - gradient_bicubic);
//      //YChannel = YChannel + 900*YChannel.mul(erodedExtractedEdges - gradient_bicubic,1);
//      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1) - 0.002*originalDiff;
//
//      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1);
//      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1);
//
//      //YChannel = YChannel - 0.4*originalDiff;
//      //
//      //YChannel = YChannel + YChannel.absdiff(originalDiff,1);
//      //YChannel = YChannel + YChannel.mul(graDiff,1);
//
//
//      //YChannel = YChannel + 0.2*originalDiff;
//
//      //YChannel = YChannel + (erodedExtractedEdges - gradient_bicubic);
//      
//      // gradient compensated sharp edges.
//      //YChannel = YChannel + 400*(erodedExtractedEdges - gradient_bicubic);
//
//      
//      // NOTE: get difference currently modifies the YChannel result!
//      originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
//      gradient_bicubic = getGradient(YChannel);
//      i++;
//    }
//
//#if SHOW_IMAGES
//    cv::imshow("erod-grad",(erodedExtractedEdges - gradient_bicubic));
//    cv::imshow("grad-erod",(gradient_bicubic - erodedExtractedEdges));
//    cv::imshow("YchannelFinal",YChannel);
//    cv::imshow("graDiff",graDiff);
//    cv::imshow("gradient_last_iteration",gradient_bicubic);
//    cv::imshow("originalDiff",originalDiff);
//    cv::imshow("grad*Y",YChannel.mul(graDiff,0.5));
//    //cv::subtract(YChannel,originalDiff,YChannel);
//    //cv::imshow("sum Y - OD",YChannel);
//#endif
//    
//    
//    
//
//    //cv::GaussianBlur(YChannel,YChannel,cv::Size(11,11),1,1); 
//    //cv::imshow("YChannel1d blur", YChannel);
//    //Assign modified Y channel back to vector
//    //YUVChannels[Y] = YChannel;  
//    YUVChannels[Y] = YChannel; 
//    
//   getFinalTexture(originalYChannel,YChannel,ALD,scaleFactor);
//    // convert back to RGB format.
//    cv::Mat processedRGB = convertToRGB(YUVChannels);
//#if SHOW_IMAGES
//    cv::imshow("processedRGB",processedRGB);
//#endif
//
//   // std::cout << "bicubic PSNR: " << getPSNR(inputImage,inputBICUBIC) << "\n";
//    std::cout << "bicubic PSNR: " << getPSNR(inputImage,RGBOrg) << "\n";
//    std::cout << "sharpened PSNR: " << getPSNR(inputImage,processedRGB) << "\n";
  }

  void testSharpenEdges5(char* scaleFactorString)
  {
    scaleFactor = std::atof(scaleFactorString);
    cv::Size outImageSize(0,0);

    // grab original Y channel.
    cv::Mat inputImage = cv::imread(input);
    if (inputImage.rows == 0) std::cout << "NOPE\n";
    
    // downscale original so we can test the uprezzed.
    cv::Mat inputDownscaled;
    cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    //cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);

    // upscale bicubic downscaled image using bicubic.
    cv::Mat inputBICUBIC;
    cv::resize(inputDownscaled,inputBICUBIC,outImageSize,scaleFactor,scaleFactor,CV_INTER_CUBIC);

    cv::Mat YUVIn = inputDownscaled.clone();
    std::vector<cv::Mat> originalYUVChannels = convertToYUV(YUVIn);
    cv::Mat originalYChannel = originalYUVChannels[Y];
    // input bicubic is real resized bicubic?????????????????

    // Try straight conversion to YUV->RGB and compare PSNR.
    // Turns out PSNR decreases slightly when converting from RGB->YUV-RGB. Due to opencv bug.
    cv::Mat BY = inputBICUBIC.clone();
    std::vector<cv::Mat> BYY = convertToYUV(BY);
    cv::Mat RGBOrg = convertToRGB(BYY);


   
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
    

    cv::Mat original_gradient;
    cv::Mat gradient_bicubic;
    gradient_bicubic = getGradient(ALD.clone());
#if SHOW_IMAGES
    cv::imshow("Gradient ALD",gradient_bicubic);
#endif
    // G_b = gradient_bicubic.
    gradient_bicubic = getGradient(YChannel.clone());
    original_gradient = getGradient(BYY[Y]);
#if SHOW_IMAGES
    cv::imshow("Gradient bicubic",gradient_bicubic);
    cv::imshow("Gradient original",original_gradient);
#endif
 

    cv::Size blurKernel = cv::Size(3,3);

    // Extract edges from ALD.
    cv::Mat extractMask = ALD.clone();
    cv::Mat extractedEdges;
    extractedEdges = getGradient(YChannel);
    
    cv::Scalar threshold = cv::mean(ALD);
    float avgALD = 1.5 * threshold.val[0];

    cv::Mat gradient_b_dilated;
    cv::dilate(gradient_bicubic,gradient_b_dilated,cv::Mat());
    cv::imshow("gbd",gradient_b_dilated);


    //for (int y = 0; y < gradient_bicubic.rows; y++) 
    //{
    //  for (int x = 0; x < gradient_bicubic.cols; x++) 
    //  {
    //    //grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
    //    if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  20)
    //    {
    //      extractMask.at<unsigned char>(y,x) = 255;
    //    }
    //    else 
    //    {
    //      extractMask.at<unsigned char>(y,x) = 0;
    //    }
    //  }
    //}
    //imshow("goodGradMap", extractMask);



    cv::Mat maskedY = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat extractALD = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    for (int y = 0; y < gradient_bicubic.rows; y++) 
    {
      for (int x = 0; x < gradient_bicubic.cols; x++) 
      {
	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
	if ((int)ALD.at<unsigned char>(y,x) >  avgALD)
	//if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  20)
	{
          //extractALD.at<unsigned char>(y,x) = 255;
	  maskedY.at<unsigned char>(y,x) = gradient_bicubic.at<unsigned char>(y,x);
	}
	else 
	{
          extractALD.at<unsigned char>(y,x) = 0;
	}
      }
    }


    cv::imshow("extractALD",extractALD);
    //cv::multiply(YChannel.clone(),extractALD,maskedY,1);
    //YChannel.copyTo(maskedY,extractALD);
    cv::imshow("maskedY",maskedY);
    
//    cv::Mat gradMaskY = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
//    gradMaskY = getGradient(maskedY);
//    cv::imshow("gradMaskY",gradMaskY);
    cv::Mat gradMaskY = maskedY.clone();
    cv::imshow("gradMaskY",gradMaskY);



    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
    cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),0,1);
    cv::Mat erodedMaskY;
    cv::imshow("gradMaskpreero",gradMaskY);
    cv::erode(gradMaskY,erodedMaskY,cv::Mat(),cv::Point(-1,-1),1);
    cv::imshow("erodedMaskedY",erodedMaskY);
    cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),1,0);
    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    gradMaskY *= 2;
    cv::imshow("erodedMaskedFinal",gradMaskY);

    

    cv::imshow("YchanelnoBlur",YChannel);
    cv::Mat testblur = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::GaussianBlur(YChannel,testblur,cv::Size(25,25),0,9);
    cv::imshow("testblurY",testblur);
    cv::GaussianBlur(YChannel,testblur,cv::Size(25,25),9,0);
    cv::imshow("testblurX",testblur);

    //cv::Mat gradEroded  = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    //gradEroded = getGradient(YChannel);

    cv::Mat Y2;
    cv::Mat Xt = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat XtD = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat diff = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gradB = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gb2 = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gh2 = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    Y2 = originalYChannel.clone(); 
    Xt = YChannel.clone(); 
    XtD = YChannel.clone(); 
    gradB = getGradient(Xt);
     



    int iterator = 0;
    while (iterator < 50)
    {
      cv::GaussianBlur(Xt,XtD,cv::Size(5,5),1,0);
      cv::resize(XtD,XtD,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
      diff = Y2 - XtD;
      cv::resize(diff,diff,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
      cv::GaussianBlur(diff,diff,cv::Size(5,5),0,1);

      gradB = getGradient(Xt);
      cv::pow(gradB,2,gb2);
      cv::pow(gradMaskY,2,gh2);

      Xt = Xt + 0.2*diff + 0.004*(gradMaskY - gradB); 
      iterator++;
    }

   imshow("Xt",Xt);
   //YUVChannels[Y] = Xt; 
   //YUVChannels[Y] = YChannel; 
    
   //getFinalTexture(originalYChannel,YChannel,ALD,scaleFactor);
   
   cv::Mat finalImage = getFinalTexture(originalYChannel,Xt,ALD,scaleFactor);
   ////cv::Mat finalImage = getFinalTexture(originalYChannel,YChannel,ALD,scaleFactor);
   YUVChannels[Y] = finalImage; 
  // imshow("Final iamge",finalImage);

    // convert back to RGB format.
   cv::Mat processedRGB = convertToRGB(YUVChannels);
   std::cout << "Orgbicubic PSNR: " << getPSNR(inputImage,RGBOrg) << "\n";
   std::cout << "Orgsharpened PSNR: " << getPSNR(inputImage,processedRGB) << "\n";
  
//
//    for (int y = 0; y < gradient_bicubic.rows; y++) 
//    {
//      for (int x = 0; x < gradient_bicubic.cols; x++) 
//      {
//	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
//	if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  20)
//	{
//          extractMask.at<unsigned char>(y,x) = 255;
//	}
//	else 
//	{
//          extractMask.at<unsigned char>(y,x) = 0;
//	}
//      }
//    }
//
//#if SHOW_IMAGES
//    cv::imshow("extractMask",extractMask);
//#endif
//    for (int y = 0; y < gradient_bicubic.rows; y++) 
//    {
//      for (int x = 0; x < gradient_bicubic.cols; x++) 
//      {
//	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
//	if ((int)extractMask.at<unsigned char>(y,x) <  10)
//	{
//          //gradient_bicubic.at<unsigned char>(y,x) = 0;
//          extractedEdges.at<unsigned char>(y,x) = 0;
//        }
//      }
//    }
//
//       
//
//#if SHOW_IMAGES
//    cv::imshow("extractedEdges",extractedEdges);
//#endif
//
//    // Resize image and blur it.
//    cv::Size imageSize(0,0);
//    cv::resize(extractedEdges,extractedEdges,imageSize,scaleFactor,scaleFactor,CV_INTER_NN);
//    //cv::blur(extractedEdges, extractedEdges, blurKernel);
//    //cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
//    extractedEdges = sharpen(extractedEdges);
//#if SHOW_IMAGES
//    cv::imshow("extractedEdges_upscaled",extractedEdges);
//#endif
//
//    // Apply erosion operator.
//    cv::Mat erodedExtractedEdges;
//    cv::erode(extractedEdges,erodedExtractedEdges,cv::Mat(),cv::Point(-1,-1),1);
//#if SHOW_IMAGES
//    cv::imshow("erodedExtractedEdges",erodedExtractedEdges);
//#endif
//    // Reblur and downsample. 
//    //cv::blur(erodedExtractedEdges, erodedExtractedEdges, blurKernel);
//    //cv::GaussianBlur(extractedEdges,extractedEdges,blurKernel,1,1);
//    //cv::GaussianBlur(erodedExtractedEdges,erodedExtractedEdges,cv::Size(11,11),1,1);
//    //erodedExtractedEdges = sharpen(erodedExtractedEdges);
//    erodedExtractedEdges *= 5000.0;  
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp", erodedExtractedEdges);
//#endif
//	   
//    cv::resize(erodedExtractedEdges,erodedExtractedEdges,imageSize,1/scaleFactor,1/scaleFactor,CV_INTER_CUBIC);
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp_downscaled", erodedExtractedEdges);
//#endif
//   
//    erodedExtractedEdges *= 1.0;  
//#if SHOW_IMAGES
//    cv::imshow("erodedEE_sharp_downscaled * 2", erodedExtractedEdges);
//#endif
//    // G_hat = erodedExtractedEdges.
//
//#if SHOW_IMAGES
//      //cv::imshow("gb sharp", sharpened);
//      //cv::imshow("low contrast", lowContrastMask);
//#endif
//      //gradient_bicubic = sharpened;
//
//
//    cv::Mat showb = gradient_bicubic.clone();
//    cv::Mat showe = erodedExtractedEdges.clone();
//    cv::Mat showMinus = erodedExtractedEdges.clone();
//
//    cv::pow(showe,2,showe);
//    cv::pow(showb,2,showb);
//
//    //showMinus = (erodedExtractedEdges - showb);
//    //cv::subtract(showb, erodedExtractedEdges,showMinus);
//    
//  //  for (int y = 0; y < gradient_bicubic.rows; y++) 
//  //  {
//  //    for (int x = 0; x < gradient_bicubic.cols; x++) 
//  //    {
//  //      //grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
//  //      showMinus.at<unsigned char>(y,x) = 100;
//  //      //erodedExtractedEdges.at<unsigned char>(y,x) = 0;
//  //      //erodedExtractedEdges.at<unsigned char>(y,x) - 
//  //      //gradient_bicubic.at<unsigned char>(y,x);
//  //      //std::cout << ".";	
//  //    }
//  //  }
//    
//    cv::absdiff(showe,showb,showMinus);
//    //showMinus*=50;
//#if SHOW_IMAGES
//    cv::imshow("Gradient - GradientMap",showe - showb);
//    cv::imshow("Gradient - GradientMap2",showMinus);
//    cv::imshow("Gradient extract^2",showe);
//    cv::imshow("Gradient org^2",showb);
//    //imshow("Gradient*Y",showMinus*YChannel.clone());
//#endif
//
//    cv::Mat originalDiff;  
//    cv::Mat graDiff;  
//
//    cv::pow(erodedExtractedEdges,2,erodedExtractedEdges);
//    
//    // Iterative formula
//    int i = 0;
//    while (i < 50)
//    {
//      //originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
//     // pow(erodedExtractedEdges,2,erodedExtractedEdges);
//      pow(gradient_bicubic,2,gradient_bicubic);
//      cv::absdiff(erodedExtractedEdges,gradient_bicubic,graDiff);
//      cv::imshow("gb-gh", gradient_bicubic - erodedExtractedEdges);
//      cv::imshow("gh-gb", erodedExtractedEdges - gradient_bicubic);
//
//      //YChannel = YChannel;
//      //YChannel = YChannel + 0.2*originalDiff + 0.004*(graDiff);
//      //YChannel = YChannel + 0.2*originalDiff + 0.004*(graDiff);
//      //YChannel = YChannel + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient_bicubic);
//      //YChannel = YChannel + 40.4*(erodedExtractedEdges - gradient_bicubic);
//      //YChannel = YChannel + 0.20*originalDiff  + 0.004*(erodedExtractedEdges - gradient_bicubic);
//      //YChannel = YChannel + 900*YChannel.mul(erodedExtractedEdges - gradient_bicubic,1);
//      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1) - 0.002*originalDiff;
//
//      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1);
//      //YChannel = YChannel + 0.002*YChannel.mul(graDiff,1);
//
//      //YChannel = YChannel - 0.4*originalDiff;
//      //
//      //YChannel = YChannel + YChannel.absdiff(originalDiff,1);
//      //YChannel = YChannel + YChannel.mul(graDiff,1);
//
//
//      //YChannel = YChannel + 0.2*originalDiff;
//
//      //YChannel = YChannel + (erodedExtractedEdges - gradient_bicubic);
//      
//      // gradient compensated sharp edges.
//      //YChannel = YChannel + 400*(erodedExtractedEdges - gradient_bicubic);
//
//      
//      // NOTE: get difference currently modifies the YChannel result!
//      originalDiff = getDifference(originalYChannel, YChannel, blurKernel);
//      gradient_bicubic = getGradient(YChannel);
//      i++;
//    }
//
//#if SHOW_IMAGES
//    cv::imshow("erod-grad",(erodedExtractedEdges - gradient_bicubic));
//    cv::imshow("grad-erod",(gradient_bicubic - erodedExtractedEdges));
//    cv::imshow("YchannelFinal",YChannel);
//    cv::imshow("graDiff",graDiff);
//    cv::imshow("gradient_last_iteration",gradient_bicubic);
//    cv::imshow("originalDiff",originalDiff);
//    cv::imshow("grad*Y",YChannel.mul(graDiff,0.5));
//    //cv::subtract(YChannel,originalDiff,YChannel);
//    //cv::imshow("sum Y - OD",YChannel);
//#endif
//    
//    
//    
//
//    //cv::GaussianBlur(YChannel,YChannel,cv::Size(11,11),1,1); 
//    //cv::imshow("YChannel1d blur", YChannel);
//    //Assign modified Y channel back to vector
//    //YUVChannels[Y] = YChannel;  
//    YUVChannels[Y] = YChannel; 
//    
//   getFinalTexture(originalYChannel,YChannel,ALD,scaleFactor);
//    // convert back to RGB format.
//    cv::Mat processedRGB = convertToRGB(YUVChannels);
//#if SHOW_IMAGES
//    cv::imshow("processedRGB",processedRGB);
//#endif
//
//   // std::cout << "bicubic PSNR: " << getPSNR(inputImage,inputBICUBIC) << "\n";
//    std::cout << "bicubic PSNR: " << getPSNR(inputImage,RGBOrg) << "\n";
//    std::cout << "sharpened PSNR: " << getPSNR(inputImage,processedRGB) << "\n";
  }

  void testSharpenEdges6(char* imageName, char* scaleFactorString)
  {
    scaleFactor = std::atof(scaleFactorString);
    cv::Size outImageSize(0,0);

    // grab original Y channel.
    //cv::Mat inputImage = cv::imread(input);
    cv::Mat inputImage = cv::imread(imageName);
    if (inputImage.rows == 0) std::cout << "NOPE\n";
    


cv::Mat YUVInOrg = inputImage.clone();
std::vector<cv::Mat> inputYUVOrg = convertToYUV(YUVInOrg);
cv::Mat originalYChannelOrg = inputYUVOrg[Y];
imshow("orgo",originalYChannelOrg);






    // downscale original so we can test the uprezzed.
    //cv::Mat inputDownscaled;
    cv::Mat inputDownscaled = inputImage.clone();
    //cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    cv::resize(inputImage,inputDownscaled,outImageSize,1.0,1.0,CV_INTER_CUBIC);
    //cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);

    // upscale bicubic downscaled image using bicubic.
    cv::Mat inputBICUBIC;
    cv::resize(inputDownscaled,inputBICUBIC,outImageSize,scaleFactor,scaleFactor,CV_INTER_CUBIC);
    cv::imshow("RGB_BI",inputBICUBIC);

    cv::Mat YUVIn = inputDownscaled.clone();
    std::vector<cv::Mat> originalYUVChannels = convertToYUV(YUVIn);
    cv::Mat originalYChannel = originalYUVChannels[Y];
    // input bicubic is real resized bicubic?????????????????

    // Try straight conversion to YUV->RGB and compare PSNR.
    // Turns out PSNR decreases slightly when converting from RGB->YUV-RGB. Due to opencv bug.
    cv::Mat BY = inputBICUBIC.clone();
    std::vector<cv::Mat> BYY = convertToYUV(BY);
    cv::Mat RGBOrg = convertToRGB(BYY);


   
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
    

    cv::Mat original_gradient;
    cv::Mat gradient_bicubic;
    gradient_bicubic = getGradient(ALD.clone());
#if SHOW_IMAGES
    cv::imshow("Gradient ALD",gradient_bicubic);
#endif
    // G_b = gradient_bicubic.
    gradient_bicubic = getGradient(YChannel.clone());
    original_gradient = getGradient(BYY[Y]);
#if SHOW_IMAGES
    cv::imshow("Gradient bicubic",gradient_bicubic);
    cv::imshow("Gradient original",original_gradient);
#endif
 

    cv::Size blurKernel = cv::Size(3,3);

    // Extract edges from ALD.
    cv::Mat extractMask = ALD.clone();
    cv::Mat extractedEdges;
    extractedEdges = getGradient(YChannel);
    
    cv::Scalar threshold = cv::mean(ALD);
    float avgALD = 1.5 * threshold.val[0];

    cv::Mat gradient_b_dilated;
    cv::dilate(gradient_bicubic,gradient_b_dilated,cv::Mat());
    cv::imshow("gbd",gradient_b_dilated);


    //for (int y = 0; y < gradient_bicubic.rows; y++) 
    //{
    //  for (int x = 0; x < gradient_bicubic.cols; x++) 
    //  {
    //    //grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
    //    if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  20)
    //    {
    //      extractMask.at<unsigned char>(y,x) = 255;
    //    }
    //    else 
    //    {
    //      extractMask.at<unsigned char>(y,x) = 0;
    //    }
    //  }
    //}
    //imshow("goodGradMap", extractMask);

    imshow("gb33",gradient_bicubic);

    cv::Mat maskedY = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat extractALD = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    for (int y = 0; y < gradient_bicubic.rows; y++) 
    {
      for (int x = 0; x < gradient_bicubic.cols; x++) 
      {
	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
	//if ((int)ALD.at<unsigned char>(y,x) >  avgALD)
	if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  0)
	{
          //extractALD.at<unsigned char>(y,x) = 255;
	  maskedY.at<unsigned char>(y,x) = gradient_bicubic.at<unsigned char>(y,x);
	}
	else 
	{
          extractALD.at<unsigned char>(y,x) = 0;
	}
      }
    }


    cv::imshow("extractALD",extractALD);
    //cv::multiply(YChannel.clone(),extractALD,maskedY,1);
    //YChannel.copyTo(maskedY,extractALD);
    cv::imshow("maskedY",maskedY);
    
//    cv::Mat gradMaskY = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
//    gradMaskY = getGradient(maskedY);
//    cv::imshow("gradMaskY",gradMaskY);
    cv::Mat gradMaskY = maskedY.clone();
    cv::imshow("gradMaskY",gradMaskY);



//    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
//    cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),0,1);
//    cv::Mat erodedMaskY;
//    cv::imshow("gradMaskpreero",gradMaskY);
//    cv::erode(gradMaskY,erodedMaskY,cv::Mat(),cv::Point(-1,-1),1);
//    cv::imshow("erodedMaskedY",erodedMaskY);
//    cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),1,0);
//    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
//    gradMaskY *= 2;
//    cv::imshow("erodedMaskedFinal",gradMaskY);
    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
    //cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),1,1);
    cv::Mat erodedMaskY;
    gradMaskY = sharpen(gradMaskY);
    cv::imshow("gradMaskpreero",gradMaskY);
    cv::erode(gradMaskY,erodedMaskY,cv::Mat(),cv::Point(-1,-1),1);
    cv::imshow("erodedMaskedY",erodedMaskY);
    cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),1,1);
    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    //gradMaskY *= 2;
    cv::imshow("erodedMaskedFinal",gradMaskY);

    cv::GaussianBlur(erodedMaskY,erodedMaskY,cv::Size(5,5),1,1);
    cv::resize(erodedMaskY,erodedMaskY,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    erodedMaskY *= 3;
    cv::imshow("erodedMaskedFinal",erodedMaskY);


    //pass 2!!!!!!!!
  //  cv::Mat gradMaskY2 = erodedMaskY.clone(); 
  //  cv::Mat erodedMaskY2; 
  //  cv::resize(gradMaskY2,gradMaskY2,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
  //  //cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),1,1);
  //  gradMaskY2 = sharpen(gradMaskY2);
  //  cv::imshow("gradMaskpreero2",gradMaskY2);
  //  cv::erode(gradMaskY2,erodedMaskY2,cv::Mat(),cv::Point(-1,-1),1);
  //  cv::imshow("erodedMaskedY2",erodedMaskY2);
  //  cv::GaussianBlur(gradMaskY2,gradMaskY2,cv::Size(5,5),1,1);
  //  cv::resize(gradMaskY2,gradMaskY2,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
  //  //gradMaskY *= 2;
  //  cv::imshow("erodedMaskedFinal2",erodedMaskY2);

  //  cv::GaussianBlur(erodedMaskY2,erodedMaskY2,cv::Size(5,5),1,1);
  //  cv::resize(erodedMaskY2,erodedMaskY2,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
  //  erodedMaskY2 *= 2;
  //  cv::imshow("erodedMaskedFinal2",erodedMaskY2);


    

    //cv::imshow("YchanelnoBlur",YChannel);
    //cv::Mat testblur = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    //cv::GaussianBlur(YChannel,testblur,cv::Size(25,25),0,9);
    //cv::imshow("testblurY",testblur);
    //cv::GaussianBlur(YChannel,testblur,cv::Size(25,25),9,0);
    //cv::imshow("testblurX",testblur);

    //cv::Mat gradEroded  = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    //gradEroded = getGradient(YChannel);


    cv::Mat resizedEMY = erodedMaskY.clone();
    cv::Mat resizedY = YChannel.clone();
    cv::resize(resizedEMY,resizedEMY,cv::Size(0,0),4.0,4.0,CV_INTER_CUBIC);
    //cv::GaussianBlur(resizedEMY,resizedEMY,cv::Size(25,25),9,9);
    cv::resize(resizedY,resizedY,cv::Size(0,0),4.0,4.0,CV_INTER_CUBIC);
    cv::Mat sharpY = resizedY.clone();
    //cv::Mat sharpY = YChannel.clone();

    int tRadius = 10;
    int tThreshold = 1;
    int tMask = 25;
    //int tMask = 110;

    gradientSharpen(resizedEMY,resizedY,sharpY,tRadius,tThreshold,tMask);
    //gradientSharpen(erodedMaskY,YChannel.clone(),sharpY,tRadius,tThreshold,tMask);
    cv::imshow("YC_SUPER",sharpY);
    cv::Mat sharpMask = sharpY - resizedY;
    //cv::Mat sharpMask = sharpY - YChannel.clone();
    cv::imshow("YC_SUPER_mask",sharpMask);
    cv::Mat sharpCanvas = resizedY.clone();
    //cv::Mat sharpCanvas = YChannel.clone();
    sharpCanvas = 128;
    gradientSharpen(resizedEMY,resizedY,sharpCanvas,tRadius,tThreshold,tMask);
    //gradientSharpen(erodedMaskY,YChannel.clone(),sharpCanvas,tRadius,tThreshold,tMask);
    cv::imshow("YC_SUPER_canvas",sharpCanvas);
    //cv::GaussianBlur(sharpCanvas,sharpCanvas,cv::Size(3,3),1,1);
    //cv::resize(sharpCanvas,sharpCanvas,cv::Size(0,0),2.0,2.0,CV_INTER_CUBIC);
    //cv::resize(sharpCanvas,sharpCanvas,cv::Size(0,0),0.5,0.5,CV_INTER_CUBIC);
    //imshow("YC_SUPER_canvas_blurred",sharpCanvas);

    cv::resize(sharpY,sharpY,cv::Size(0,0),0.25,0.25,CV_INTER_CUBIC);
    cv::resize(sharpCanvas,sharpCanvas,cv::Size(0,0),0.25,0.25,CV_INTER_CUBIC);

    cv::imshow("YC_SUPER_r",sharpY);
    cv::imshow("YC_SUPER_canvas_r",sharpCanvas);

    cv::Mat Y2;
    cv::Mat Xt = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat XtD = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat diff = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gradB = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
   // cv::Mat gb2 = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
   // cv::Mat gh2 = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gb2 = gradB.clone();
    gb2.convertTo(gb2,CV_32FC1);
    cv::Mat gh2 = erodedMaskY.clone();
    gh2.convertTo(gh2,CV_32FC1);
    cv::Mat subghgb = cv::Mat::zeros(ALD.rows,ALD.cols,CV_32F);
    cv::Mat gg = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    Y2 = originalYChannel.clone(); 
    Xt = YChannel.clone(); 
    XtD = YChannel.clone(); 
    gradB = getGradient(Xt);
     
   cv::imshow("gradb",gradB);
   cv::imshow("subt",erodedMaskY - gradB);

   for (int y = 0;y<gradB.rows;y++)
   {
     for (int y = 0;y<gradB.rows;y++)
     {
        
     }
   }

    int iterator = 0;
    while (iterator < 1)
    {
      cv::GaussianBlur(Xt,XtD,cv::Size(5,5),1,0);
      cv::resize(XtD,XtD,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
      diff = Y2 - XtD;
      cv::resize(diff,diff,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
      cv::GaussianBlur(diff,diff,cv::Size(5,5),0,1);

      //gradB = getGradient(Xt);
      //cv::pow(gb2,2,gb2);
      //cv::pow(gh2,2,gh2);
      ////cv::pow(erodedMaskY,2,gh2);

      ////cv::pow(gradB,2,gb2);
      ////cv::pow(gradMaskY,2,gh2);
      //////cv::pow(erodedMaskY,2,gh2);

      //imshow("gb2",gb2);
      //imshow("gh2",gh2);
      //imshow("gh2 - gb2",gh2 - gb2);

      //subghgb = gh2 - gb2;
      //cv::pow(subghgb,2,subghgb);
      //imshow("subg",subghgb);
      //gg = (erodedMaskY - gradB);

      

     // cv::normalize(erodedMaskY,erodedMaskY,1,1,cv::NORM_MINMAX);
     // cv::normalize(gradB,gradB,1,1,cv::NORM_MINMAX);
       
     
     // use tier of gradients with different masking values
     // and different multipliers to properly recover
     // hard vs soft edges.
     // dilate eroded gradient, threshold dilated gradient,
     // subtract eroded from dilated, use apply new dilated 
     // (split gggggg

     

     cv::Mat subg = erodedMaskY - gradB;
     cv::Scalar mean = cv::mean(Xt);
     float xtMean = mean.val[0];

     float lambda1 = 1.0;
     float lambda2 = 0.7;

     for (int y = 0;y<gradB.rows;y++)
     {
       for (int x = 0;x<gradB.rows;x++)
       {
         int currentValue = Xt.at<unsigned char>(y,x);
         int modifier = 0.02*(subg.at<unsigned char>(y,x) * subg.at<unsigned char>(y,x));
         if (xtMean < 256 /*127*/) 
	 {
		 if ( currentValue >= modifier )
                 {
	           Xt.at<unsigned char>(y,x) -= lambda2*modifier;
		 }
		 else
		 {
	           Xt.at<unsigned char>(y,x) = 0;
		 }
	 }
	 else
	 {
                 if ( currentValue + modifier <= 255)
                 {
	           Xt.at<unsigned char>(y,x) += modifier;
		 }
		 else
		 {
	           Xt.at<unsigned char>(y,x) = 255;
		 }
	 }
       }
     }


     //Xt += 2.9*diff; 
     Xt += lambda1*diff; 


      //Xt = Xt + 0.2*diff + 0.004*(gh2 -  gb2); 

      //std::cout << "Mean: " << xtMean << std::endl;
     // if (xtMean > 127) Xt = Xt /*+ 0.2*diff*/ + 1*(erodedMaskY - gradB); 
     // else  Xt = Xt /*+ 0.2*diff*/ - 1*(erodedMaskY - gradB); 
      //Xt = Xt /*+ 0.2*diff*/ + 1*subghgb ;
      iterator++;
    }

    imshow("diff",diff);
   imshow("Xt",Xt);
    cv::pow(gg,2,gg);
    cv::imshow("lines",gg);
   //YUVChannels[Y] = Xt;  ///////////////////////////////
   YUVChannels[Y] = sharpY; 
   //YUVChannels[Y] = YChannel; 
    
   //cv::Mat finalImage = getFinalTexture(originalYChannel,Xt,ALD,scaleFactor);
   //YUVChannels[Y] = finalImage; 
   //imshow("Final iamge",finalImage);

    // convert back to RGB format.
   cv::Mat processedRGB = convertToRGB(YUVChannels);
 
   cv::imshow("RGB",processedRGB);
   cv::resize(RGBOrg,RGBOrg,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_NN);
   cv::resize(processedRGB,processedRGB,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_NN);
   cv::resize(sharpY,sharpY,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_NN);

   std::cout << "Orgbicubic PSNR: " << getPSNR(inputImage,RGBOrg) << "\n";
   std::cout << "Orgsharpened PSNR: " << getPSNR(inputImage,processedRGB) << "\n";

   //std::cout << "Orgbicubic PSNR: " << getPSNR(originalYChannelOrg,YChannel) << "\n";
   //std::cout << "Orgsharpened PSNR: " << getPSNR(originalYChannelOrg,sharpY) << "\n";
  }


  void testPSNR()
  {
    double avgPSNR = 0;
    cv::Size outImageSize(0,0);

    for (unsigned long int i = 0; i < outFrames.size(); i++ ) 
    {
      avgPSNR += getPSNR(inFrames[i],outFrames[i]);
    }
    avgPSNR /= outFrames.size();

    std::cout << "Average PSNR per frame: " << avgPSNR << std::endl;
  }

  void testPSNRDownScale()
  {
    double avgPSNR = 0;
    cv::Size outImageSize(0,0);

    for (unsigned long int i = 0; i < outFrames.size(); i++ ) 
    {
      // Downscale the output to the original image size, 
      // and do PSNR against original image and downscaled output image.
      cv::resize(outFrames[i],outFrames[i],outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_NN);
      avgPSNR += getPSNR(inFrames[i],outFrames[i]);
    }
    avgPSNR /= inFrames.size();

    std::cout << "Average PSNR per frame: " << avgPSNR << std::endl;
  }

private:

  double scaleFactor;
  std::vector<cv::Mat> inFrames;
  std::vector<cv::Mat> orgFrames;
  std::vector<cv::Mat> outFrames;
  cv::Mat ALD;
  cv::VideoCapture capture;

 float distance(float x1, float y1, float x2, float y2)
  {
    return cv::sqrt( (x1-x2)*(x1-x2) - (y1-y2)*(y1-y2) );
  } 
 
  // Divides gradient map in half along gradient curves and sharpens each half.
  void splitSharpen(cv::Mat gmap, cv::Mat ymap, cv::Mat sharpenedYmap, cv::Point p, int maxRadius, int threshold)
  {
    int radius = 1;
    int difference = 0;
    int maxDiff = 0;
    int maxDiffx = -1;
    int maxDiffy = -1;

    while (radius <= maxRadius)
    {
      for (int y = (p.y - radius); y <= (p.y + radius) && (y >= 0) && (y < gmap.rows) ; y++)
      {
        for (int x = (p.x - radius); x <= (p.x + radius) && (x >= 0) && (x < gmap.cols) ; x++)
        {
            if ( distance(p.x, p.y, x, y) <= radius /* && 
          		(x >= 0) && 
          		(y >= 0) &&
          		(x < gmap.cols) &&
          		(y < gmap.rows) */ )
            {
              	// dont actually need all of the above.
              	// can search for closest pixel using a
              	// winding pattern going around the center
              	// pixel, and stopping search once conditions are met.
              	// Will be more efficient.
              	//
              	// Algo summary:
              	// 1) Take difference of a thresholded 
              	// non-black pixel and its
              	// closest neighbors in gradient map.
              	// 2) Choose neighbor with difference greater 
              	// than a threshold, and assign it's Y value 
              	// to the original pixel.
              	// 
              	// Effect: edge contours will be sharpened 
              	// along their centers, instead of along 
              	// the outside edge.
              	// 
              	// Can search for pixels using multiple radii,
              	// ie. search all pixels for radius = 1, 
              	// store the max difference and pixel where max occurs,
              	// if difference < threshold, continue
              	// else assign the YValue of max diff pixel to current pixel.
              
              
              //difference = cv::fast_abs( (int)gmap.at<unsigned char>(p.y,p.x) - (int)gmap.at<unsigned char>(y,x) );
              difference = (int)gmap.at<unsigned char>(p.y,p.x) - (int)gmap.at<unsigned char>(y,x);
	      //if (difference <= 0) difference = 0; 
	      
	      if ( difference > maxDiff )
	      {
                maxDiff = difference;
		maxDiffx = x;
		maxDiffy = y;
	      }
            }
          }
        }
      if (maxDiff >= threshold)
      {
        sharpenedYmap.at<unsigned char>(p.y,p.x) = (int)ymap.at<unsigned char>(maxDiffy,maxDiffx);
	break;
      }
      radius++;
    }
  }

  void gradientSharpen(cv::Mat gmap, cv::Mat ymap, cv::Mat sharpenedYmap, int radius, int threshold, int maskValue)
  {
    for (int y = 0; y < gmap.rows; y++)
    {
      for (int x = 0; x < gmap.cols; x++)
      {
        //ALD.at<unsigned char>(y,x) = (int)calculateALD(ALD, cv::Point(x,y), radius);
	if ((int)gmap.at<unsigned char>(y,x) > maskValue )
	{
	  splitSharpen(gmap, ymap, sharpenedYmap,cv::Point(x,y),radius, threshold);
	}
      }
    }
  }


  void sharpenEdgesOrg(int i)
  {
    cv::Mat inputImage = inFrames[i];
    //// scaleFactor = std::atof(scaleFactorString);
    cv::Size outImageSize(0,0);

    //// grab original Y channel.
    //cv::Mat inputImage = cv::imread(input);
    //if (inputImage.rows == 0) std::cout << "NOPE\n";
    
    // downscale original so we can test the uprezzed.
    cv::Mat inputDownscaled;
    cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    //cv::resize(inputImage,inputDownscaled,outImageSize,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);

    // upscale bicubic downscaled image using bicubic.
    cv::Mat inputBICUBIC;
    cv::resize(inputDownscaled,inputBICUBIC,outImageSize,scaleFactor,scaleFactor,CV_INTER_CUBIC);

    cv::Mat YUVIn = inputDownscaled.clone();
    std::vector<cv::Mat> originalYUVChannels = convertToYUV(YUVIn);
    cv::Mat originalYChannel = originalYUVChannels[Y];
    // input bicubic is real resized bicubic?????????????????

    // Try straight conversion to YUV->RGB and compare PSNR.
    // Turns out PSNR decreases slightly when converting from RGB->YUV-RGB. Due to opencv bug.
    cv::Mat BY = inputBICUBIC.clone();
    std::vector<cv::Mat> BYY = convertToYUV(BY);
    cv::Mat RGBOrg = convertToRGB(BYY);


   
    // grab bicubic resized Y channel.
    cv::Mat YUVOut = inputBICUBIC.clone();
#if SHOW_IMAGES
    //cv::imshow("OriginalRGB",YUVOut);
#endif
    std::vector<cv::Mat> YUVChannels = convertToYUV(YUVOut);
    ////cv::imshow("YUVOut",YUVOut);

    cv::Mat YChannel = YUVChannels[Y];
#if SHOW_IMAGES
    //cv::imshow("Ychannel", YChannel);
#endif
    // Use YChannel for rest of algorthm. 


    // Calculate ALD.
    cv::Mat YChannelCopy = YChannel.clone();
    cv::Mat ALD = getALD(YChannelCopy,ALDRadius);
#if SHOW_IMAGES
    //cv::imshow("ALD",ALD);
#endif
    //YChannel = ALD;
    

    cv::Mat original_gradient;
    cv::Mat gradient_bicubic;
    gradient_bicubic = getGradient(ALD.clone());
#if SHOW_IMAGES
    //cv::imshow("Gradient ALD",gradient_bicubic);
#endif
    // G_b = gradient_bicubic.
    gradient_bicubic = getGradient(YChannel.clone());
    original_gradient = getGradient(BYY[Y]);
#if SHOW_IMAGES
    //cv::imshow("Gradient bicubic",gradient_bicubic);
    //cv::imshow("Gradient original",original_gradient);
#endif
 

    cv::Size blurKernel = cv::Size(3,3);

    // Extract edges from ALD.
    cv::Mat extractMask = ALD.clone();
    cv::Mat extractedEdges;
    extractedEdges = getGradient(YChannel);
    
    cv::Scalar threshold = cv::mean(ALD);
    float avgALD = 1.5 * threshold.val[0];

    cv::Mat gradient_b_dilated;
    cv::dilate(gradient_bicubic,gradient_b_dilated,cv::Mat());
    //cv::imshow("gbd",gradient_b_dilated);


    //for (int y = 0; y < gradient_bicubic.rows; y++) 
    //{
    //  for (int x = 0; x < gradient_bicubic.cols; x++) 
    //  {
    //    //grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
    //    if ((int)gradient_b_dilated.at<unsigned char>(y,x) >  20)
    //    {
    //      extractMask.at<unsigned char>(y,x) = 255;
    //    }
    //    else 
    //    {
    //      extractMask.at<unsigned char>(y,x) = 0;
    //    }
    //  }
    //}
    //imshow("goodGradMap", extractMask);



    cv::Mat maskedY = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat extractALD = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    for (int y = 0; y < gradient_bicubic.rows; y++) 
    {
      for (int x = 0; x < gradient_bicubic.cols; x++) 
      {
	//grad.at<unsigned char>(y,x) = (int)img.at<unsigned char>(y,x);
	if ((int)ALD.at<unsigned char>(y,x) >  avgALD)
	{
          //extractALD.at<unsigned char>(y,x) = 255;
	  //maskedY.at<unsigned char>(y,x) = YChannel.at<unsigned char>(y,x);
	  maskedY.at<unsigned char>(y,x) = gradient_bicubic.at<unsigned char>(y,x);
	}
	else 
	{
          extractALD.at<unsigned char>(y,x) = 0;
	}
      }
    }


    //cv::imshow("extractALD",extractALD);
    //cv::multiply(YChannel.clone(),extractALD,maskedY,1);
    //YChannel.copyTo(maskedY,extractALD);
    cv::imshow("maskedY",maskedY);
    
    cv::Mat gradMaskY = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    //gradMaskY = getGradient(maskedY);
    gradMaskY = maskedY.clone();
    cv::imshow("gradMaskY",gradMaskY);

    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
    cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),0,1);
    cv::Mat erodedMaskY;
    //cv::imshow("gradMaskpreero",gradMaskY);
    cv::erode(gradMaskY,erodedMaskY,cv::Mat(),cv::Point(-1,-1),1);
    //cv::imshow("erodedMaskedY",erodedMaskY);
    cv::GaussianBlur(gradMaskY,gradMaskY,cv::Size(5,5),1,0);
    cv::resize(gradMaskY,gradMaskY,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
    gradMaskY *= 2;
    //cv::imshow("erodedMaskedFinal",gradMaskY);

    

    ////cv::imshow("YchanelnoBlur",YChannel);
    //cv::Mat testblur = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    //cv::GaussianBlur(YChannel,testblur,cv::Size(25,25),0,9);
    ////cv::imshow("testblurY",testblur);
    //cv::GaussianBlur(YChannel,testblur,cv::Size(25,25),9,0);
    ////cv::imshow("testblurX",testblur);

    ////cv::Mat gradEroded  = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    ////gradEroded = getGradient(YChannel);

    cv::Mat Y2;
    cv::Mat Xt = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat XtD = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat diff = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gradB = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gb2 = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    cv::Mat gh2 = cv::Mat::zeros(ALD.rows,ALD.cols,ALD.type());
    Y2 = originalYChannel.clone(); 
    Xt = YChannel.clone(); 
    XtD = YChannel.clone(); 
    gradB = getGradient(Xt);
     



    int iterator = 0;
    while (iterator < 50)
    {
      cv::GaussianBlur(Xt,XtD,cv::Size(5,5),1,0);
      cv::resize(XtD,XtD,cv::Size(0,0),1.0/scaleFactor,1.0/scaleFactor,CV_INTER_CUBIC);
      diff = Y2 - XtD;
      cv::resize(diff,diff,cv::Size(0,0),scaleFactor,scaleFactor,CV_INTER_CUBIC);
      cv::GaussianBlur(diff,diff,cv::Size(5,5),0,1);

      gradB = getGradient(Xt);
      cv::pow(gradB,2,gb2);
      cv::pow(gradMaskY,2,gh2);

      Xt = Xt + 0.2*diff + 0.004*(gradMaskY - gradB); 
      iterator++;
    }

   cv::imshow("Xt",Xt);
//   YUVChannels[Y] = Xt; 
   //YUVChannels[Y] = YChannel; 
    
   //getFinalTexture(originalYChannel,YChannel,ALD,scaleFactor);
   //cv::Mat finalImage = getFinalTexture(originalYChannel,Xt,ALD,scaleFactor);
   cv::Mat finalImage = getFinalTexture(originalYChannel,YChannel,ALD,scaleFactor);
   YUVChannels[Y] = finalImage; 
   //YUVChannels[Y] = Xt;; 

    // convert back to RGB format.
   cv::Mat processedRGB = convertToRGB(YUVChannels);
   //std::cout << "Orgbicubic PSNR: " << getPSNR(inputImage,RGBOrg) << "\n";
   //std::cout << "Orgsharpened PSNR: " << getPSNR(inputImage,processedRGB) << "\n";
  
   outFrames[i] = processedRGB;

  }

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
    //cv::blur(HREstimate, HREDownsampled, blurKernel);
    cv::GaussianBlur(HREstimate,HREDownsampled,blurKernel,1,1);
    cv::resize(HREDownsampled,HREDownsampled, cv::Point(0,0), 1/scaleFactor, 1/scaleFactor, CV_INTER_CUBIC);
    cv::imshow("extracted_edges_blurred",HREDownsampled);
    
    //cv::Mat originalDiff = originalYChannel - HREDownsampled;
    //cv::Mat originalDiff = original - HREDownsampled;
    cv::Mat originalDiff = cv::Mat::zeros(HREDownsampled.rows,HREDownsampled.cols,HREDownsampled.type());
    cv::Mat positiveDiff = cv::Mat::zeros(HREDownsampled.rows,HREDownsampled.cols,HREDownsampled.type());
    cv::Mat negativeDiff = cv::Mat::zeros(HREDownsampled.rows,HREDownsampled.cols,HREDownsampled.type());
    /////////////Used to retrieve absdiff, now retrieve +/- diff seperately.
    //cv::absdiff(original,HREDownsampled,originalDiff);
    cv::imshow("o-HR",5*(original - HREDownsampled) );
    cv::imshow("HR-o",5*(HREDownsampled - original) );
    //originalDiff += original - HREDownsampled;
    //cv::imshow("ordddd",originalDiff);

    positiveDiff += (original - HREDownsampled);
    imshow("pdiff",positiveDiff);
    negativeDiff += (HREDownsampled - original);
    imshow("ndiff",negativeDiff);

    //NOTE: this line modifies final Y channel.
    //HREDownsampled = HREDownsampled + .2*positiveDiff - .8*negativeDiff;
    //HREDownsampled = HREDownsampled - 0.2*negativeDiff;
    //HREDownsampled = HREDownsampled + 0.2*positiveDiff - 0.2*negativeDiff;
    
//    HREDownsampled = HREDownsampled + positiveDiff - negativeDiff;

    //HREDownsampled += (original - HREDownsampled);
    //imshow("HRE+=diff",HREDownsampled);
    //HREDownsampled -= (HREDownsampled - original);
    //imshow("HRE-=diff",HREDownsampled);


    cv::imshow("origna image",original);
    cv::imshow("od",originalDiff);

   // addDifference(original.clone(),HREDownsampled,1);
    //cv::imshow("HRED",HREDownsampled);

    // Upscale and reblur.
    cv::resize(originalDiff,originalDiff, cv::Point(0,0), scaleFactor, scaleFactor, CV_INTER_CUBIC);
    //Note this line modifies Y channel.
    //cv::resize(HREDownsampled,HREstimate, cv::Point(0,0), scaleFactor, scaleFactor, CV_INTER_CUBIC);
   // cv::blur(originalDiff, originalDiff, blurKernel);
    cv::GaussianBlur(originalDiff, originalDiff, blurKernel,1,1);
    cv::imshow("Upscaled difference with original",originalDiff);
    
    return originalDiff;
  } 

  //cv::Mat Difference(cv::Mat original, cv::Mat reconstructed)
  void addDifference(cv::Mat original, cv::Mat& reconstructed, float scale)
  {
    cv::Mat difference = original.clone();
    for (int y = 0; y < original.rows; y++) 
    {
      for (int x = 0; x < original.cols; x++) 
      {
	int signedDiff = (int)reconstructed.at<unsigned char>(y,x) - (int)original.at<unsigned char>(y,x);
	if (signedDiff >= 0)
	{
          reconstructed.at<unsigned char>(y,x) -= signedDiff*scale;
	}
	else 
	{
          reconstructed.at<unsigned char>(y,x) += signedDiff*scale;
	}
      }
    }   
    cv::imshow("changed",reconstructed);
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
    std::cout << "Usage: supervideo <input video file> <output video file> <scale Factor (decimal)> <Use bicbubic interpolation (0/1) >\n"; 
    return 0 /*exit(1)*/;
  }

  supervideo sv;
//  sv.readVideo(argv[1]);
////  sv.interpolate();
//  sv.resize(argv[3], argv[4]);
//  sv.writeVideo(argv[2]);
//  sv.testPSNR();
//  sv.testPSNR();
  sv.testSharpenEdges6(argv[1],argv[2]);


  //Wait until any key is pressed
#if SHOW_IMAGES
  cv::waitKey(0);
#endif
  std::cout << "Exiting application\n";

  return 0;
}

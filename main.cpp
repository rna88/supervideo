#include </usr/include/opencv2/opencv.hpp>
#include <bitset>

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
    cv::Mat ALD = getALD(YChannelCopy,15);
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


    // Iterative formula
    cv::Mat finalEdgeResult = YChannel + 0.2*originalDiff + 0.004*(erodedExtractedEdges - gradient);
    //cv::Mat finalEdgeResult = originalDiff + YChannel;
    cv::imshow("final",finalEdgeResult);
    //cv::imshow("ychane",YChannel);

    std::vector<cv::Mat> orgY = convertToYUV(orgFrames[i]);
    cv::Mat orgYChannel = orgY[Y].clone();
    
     
    cv::Mat textureArea = finalTexture (orgYChannel, YChannel, scaleFactor);
    

    //Assign modified Y channel back to vector
    //YUVChannels[Y] = YChannel;  
    YUVChannels[Y] = finalEdgeResult + textureArea;  

    // convert back to RGB format.
    cv::Mat RGB = convertToRGB(YUVChannels);
    cv::imshow("RGB",RGB);
    //outFrames[i] = RGB;

    //return finalEdgeResult;
  }

/*****************************************************************************************************************
********************** HIGH RESOLUTION TEXTURE SECTION ***********************************************************
*****************************************************************************************************************/
          //return value: LBP matrix
        //input: bicubic frame
        cv::Mat LBP (cv::Mat bicubicImage)
  {

        cv::Mat LBPImage = bicubicImage.clone();        //making another copy of the input (bicubic) image (to be modified and returned)
        int maxWidth = bicubicImage.rows;                        //find widith of the bicubic image
        int maxHeight = bicubicImage.cols;                        //find height of the bicubic image

        int center;                                                //center of the 3 by 3 matrix (going to set as the threshold value)
        int surroundingValues[8];        //The 3 by 3 matrix made into an array that'll be filled with 0's and 1's

        /*  0 1 2
            3   4
            5 6 7
        */
                
                //display's the width and length of the bicubic image
                //cout << "maxWidth is: " << maxWidth << endl;
                //cout << "maxHeight is: " << maxHeight << endl;

                

        //the general case (not the boarder pixels)
                //rows
        for (int i=1; i < maxHeight-1; i++){
                                //columns
                for (int j=1; j < maxWidth-1; j++){

                                        
                        center = (int)bicubicImage.at<unsigned char>(j,i);        //grabs the threshold value from the orignal matrix

                        //top left of the 3 by 3 matrix (index 0)
                        if((int)bicubicImage.at<unsigned char>(j-1,i-1) < center)
                                                        surroundingValues[0] = 0;
                        else
                                                        surroundingValues[0] = 1;

                        //index 1 of matrix
                        if ((int)bicubicImage.at<unsigned char>(j-1,i) < center)
                                                        surroundingValues[1] = 0;
                        else
                                                        surroundingValues[1] = 1;

                        //index 2 of matrix
                        if ((int)bicubicImage.at<unsigned char>(j-1,i+1) < center)
                                                        surroundingValues[2] = 0;
                        else
                                                        surroundingValues[2] = 1;

                        //index 3 of matrix
                        if ((int)bicubicImage.at<unsigned char>(j,i-1) < center)
                                                        surroundingValues[3] = 0;
                        else
                                                        surroundingValues[3] = 1;

                        //index 4 of matrix
                        if ((int)bicubicImage.at<unsigned char>(j,i+1) < center)
                                                        surroundingValues[4] = 0;
                        else
                                                        surroundingValues[4] = 1;

                        //index 5 of matrix
                        if ((int)bicubicImage.at<unsigned char>(j+1,i-1) < center)
                                                        surroundingValues[5] = 0;
                        else
                                                        surroundingValues[5] = 1;

                        //index 6 of matrix
                        if ((int)bicubicImage.at<unsigned char>(j+1,i) < center)
                                                        surroundingValues[6] = 0;
                        else
                                                        surroundingValues[6] = 1;

                        //index 7 of matrix
                        if ((int)bicubicImage.at<unsigned char>(j+1,i+1) < center)
                                                        surroundingValues[7] = 0;
                        else
                                                        surroundingValues[7] = 1;

                                                //setting the binary value to decimal value for the pixel
                        LBPImage.at<unsigned char>(j,i) = (surroundingValues[0]*pow(2,7)+surroundingValues[1]*pow(2,6)+surroundingValues[2]*pow(2,5)+surroundingValues[3]*pow(2,4)+surroundingValues[4]*pow(2,3)+surroundingValues[5]*pow(2,2)+surroundingValues[6]*pow(2,1)+surroundingValues[7]*pow(2,0));
                                }
        }
                //STILL NEED TO IMPLEMENT THE EDGE CASES

                return LBPImage;

        }

        //LRI is the original image that isn't upscaled
        //X is the intial upscaled bicubic LRI
        //scaleFactor = the original scale factor the user inputted

  cv::Mat HRLBP (cv::Mat LRI, cv::Mat X, double scaleFactor)
  {
          
        cv::Mat HiResLBP = X.clone();
        float lambda1 = 0.1;
        float lambda2 = 0.01;
        float lambdaPrime = 0.005;

        cv::Mat TX = LBP(X);
        //--------------------------------------------------------------------------

        cv::Mat HX = X.clone();        //used to create the same size matrix
	cv::blur(X,HX,cv::Size(3,3));
        //cv::GaussianBlur(X, HX, cv::Size(5,5), 5);

        cv::Size outImageSize(0,0);        //dummy variable for resize
        cv::Mat DHX = LRI.clone();        //used to create the same size matrix
        cv::resize(HX,DHX, outImageSize ,1.0/scaleFactor,1.0/scaleFactor,CV_INTER_NN);

        cv::Mat TDHX = DHX.clone();        //used to create the same size matrix
        TDHX = LBP(DHX);

        cv::Mat TY = LBP(LRI);

        cv::Mat TY_TDHX = TY - DHX;

        cv::Mat U_TY_TDHX = X.clone();        //used to create the same size matrix
        cv::resize(HX,DHX, outImageSize ,scaleFactor,scaleFactor,CV_INTER_NN);

        //----------------------------------- MAY NEED TO BLUR FIRST ----------------------------------------------------------------
        cv::Mat HU_TY_TDHX = X.clone();        //used to create the same size matrix
	cv::blur(U_TY_TDHX,HU_TY_TDHX,cv::Size(3,3));
        //cv::GaussianBlur(U_TY_TDHX, HU_TY_TDHX, cv::Size(5,5), 5);
        //cv::addWeighted(U_TY_TDHX, 1.5, HU_TY_TDHX, -0.5, 0, U_TY_TDHX);
        //cv::addWeighted(HU_TY_TDHX, 1.5, HU_TY_TDHX, -0.5, 0, U_TY_TDHX);                //use this sharpening if Blur is first used

        //cv::Mat HTU_TY_TDHX = X.clone();
        //cv::transpose(HU_TY_TDHX, HTU_TY_TDHX);

        //cv::Mat L1HTU_TY_TDHX = lambda1*HTU_TY_TDHX;
        cv::Mat L1HTU_TY_TDHX = lambda1*HU_TY_TDHX;

        //-----------------------------------------------------------------------------------------------------------------------------
        cv::Mat gamma = X.clone();
        gamma = lambdaPrime*LBP(X);

        cv::Mat X_GammaX = X + gamma;

        cv::Mat T_X_Gamma_X = LBP(X_GammaX);

        cv::Mat T_X_Gamma_X_TX = T_X_Gamma_X - TX;

        cv::Mat L2_T_X_Gamma_X_TX = lambda2*T_X_Gamma_X_TX;



        HiResLBP = TX + L1HTU_TY_TDHX + L2_T_X_Gamma_X_TX;
        
        return HiResLBP;
  }        

        //returns the coefficient Matrix of the Hi Res LBP
        //input: the HRLBP matrix
        cv::Mat coefficientMatrixOfHRLBP (cv::Mat HRLBP){
                
                cv::Mat coefficentC = HRLBP.clone();        //making another copy of the input (bicubic) image (to be modified and returned)
        int maxWidth = coefficentC.rows;                        //find widith of the bicubic image
        int maxHeight = coefficentC.cols;                        //find height of the bicubic image
                int matrixValue;

                //-------------------------------------------------------------------------------------------


                //rows
        for (int i=1; i < maxHeight; i++){
                                //columns
                for (int j=1; j < maxWidth; j++){
                                        matrixValue = (int)HRLBP.at<unsigned char>(j,i);
                                        
                                        std::bitset<8> b1(matrixValue);
                                        int value =0;
                                        for (int i = 0; i < 8; i++){
                                                if (b1[i] == 0)
                                                        value--;
                                                else
                                                        value++;
                                        }
                                        coefficentC.at<unsigned char>(j,i) = value;
                                }
                }
                return coefficentC;
        }

        cv::Mat elementWiseMultiply (cv::Mat HRLBP, cv::Mat ALD)
        {
                
                cv::Mat output = HRLBP.clone();
                
                int maxWidth = HRLBP.rows;                        //find widith of the bicubic image
        int maxHeight = HRLBP.cols;                        //find height of the bicubic image

                //std::transform(HRLBP.begin<float>(), HRLBP.end<float>(), ALD.begin<float>(), output.begin<float>(), std::multiplies<float>());

                
                //rows
        for (int i=1; i < maxHeight; i++){
                                //columns
                for (int j=1; j < maxWidth; j++){
                                        output.at<unsigned char>(j,i) = (int)HRLBP.at<unsigned char>(j,i)*(int)ALD.at<unsigned char>(j,i) ;
                                }
                }
                
                return output;
        }

        cv::Mat reconstructedHRI (cv::Mat LRI, cv::Mat HRLBP ,cv::Mat bicubicUpSample, double scaleFactor){
                
                int lambda = 1;                
                cv::Mat reconstructedHRI;
                cv::Size outImageSize(0,0);        //dummy variable for resize
                
                cv::Mat ALD = getALD(bicubicUpSample,15);
                

                cv::Mat CD = elementWiseMultiply(HRLBP, ALD);
                cv::Mat LCD = lambda*CD;
        
                cv::Mat UY;
                cv::resize(LRI,UY, outImageSize ,scaleFactor,scaleFactor,CV_INTER_NN);
                

                cv::Mat HTUY;
                HTUY = HRLBP.clone();
                cv::blur(UY,HTUY,cv::Size(3,3));
                //cv::GaussianBlur(UY, HTUY, cv::Size(5,5), 5);
                //cv::addWeighted(UY, 1.5, HTUY, -0.5, 0, UY);

//                int maxWidth = HTUY.rows;                        //find widith of the bicubic image
//        int maxHeight = HTUY.cols;                        //find height of the bicubic image
//
//
//                int maxWidthALD = LCD.rows;                        //find widith of the bicubic image
//        int maxHeightALD = LCD.cols;                        //find height of the bicubic image

                //cout << "HTUY: " << maxWidth <<        "                " << maxHeight << endl;
                //cout << "LCD: "<< maxWidthALD << "                " << maxHeightALD << endl;


                reconstructedHRI = HTUY + LCD;
                return reconstructedHRI;
                //return ALD;
        }


        cv::Mat finalTexture (cv::Mat originalImage, cv::Mat bicubicImage, double scaleFactor){

                cv::Mat LBPImage = LBP(originalImage);
                cv::imshow("LBP", LBPImage);

                cv::Mat HiResImage = HRLBP (originalImage, bicubicImage, scaleFactor);
                cv::imshow ("HR LBP", HiResImage);

                cv::Mat CoefficentMatrix = coefficientMatrixOfHRLBP(HiResImage);
                cv::imshow ("Coefficent Matrix", CoefficentMatrix);

                cv::Mat newHRI;
                newHRI = reconstructedHRI(originalImage, HiResImage, bicubicImage, scaleFactor);
                cv::imshow ("reconstructed HRI", newHRI);

                //missing equation 14

                return newHRI;

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

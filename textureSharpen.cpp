#include </usr/include/opencv2/opencv.hpp>
#include <bitset>


/*****************************************************************************************************************
********************** HIGH RESOLUTION TEXTURE SECTION ***********************************************************
*****************************************************************************************************************/
          //return value: LBP matrix
        //input: bicubic frame
//cv::Mat LBP (cv::Mat bicubicImage)
//{
//  cv::Mat LBPImage = bicubicImage.clone();        //making another copy of the input (bicubic) image (to be modified and returned)
//  int maxWidth = bicubicImage.rows;                        //find widith of the bicubic image
//  int maxHeight = bicubicImage.cols;                        //find height of the bicubic image
//  
//  int center;                                                //center of the 3 by 3 matrix (going to set as the threshold value)
//  int surroundingValues[8];        //The 3 by 3 matrix made into an array that'll be filled with 0's and 1's
//  
//  /*  0 1 2
//      3   4
//      5 6 7
//  */
//          
//          //display's the width and length of the bicubic image
//          //cout << "maxWidth is: " << maxWidth << endl;
//          //cout << "maxHeight is: " << maxHeight << endl;
//  
//  //the general case (not the boarder pixels)
//          //rows
//  for (int i=1; i < maxHeight-1; i++)
//  {
//    //columns
//    for (int j=1; j < maxWidth-1; j++)
//    {
//                  center = (int)bicubicImage.at<unsigned char>(j,i);        //grabs the threshold value from the orignal matrix
//  
//                  //top left of the 3 by 3 matrix (index 0)
//                  if((int)bicubicImage.at<unsigned char>(j-1,i-1) < center)
//                                                  surroundingValues[0] = 0;
//                  else
//                                                  surroundingValues[0] = 1;
//  
//                  //index 1 of matrix
//                  if ((int)bicubicImage.at<unsigned char>(j-1,i) < center)
//                                                  surroundingValues[1] = 0;
//                  else
//                                                  surroundingValues[1] = 1;
//  
//                  //index 2 of matrix
//                  if ((int)bicubicImage.at<unsigned char>(j-1,i+1) < center)
//                                                  surroundingValues[2] = 0;
//                  else
//                                                  surroundingValues[2] = 1;
//  
//                  //index 3 of matrix
//                  if ((int)bicubicImage.at<unsigned char>(j,i-1) < center)
//                                                  surroundingValues[3] = 0;
//                  else
//                                                  surroundingValues[3] = 1;
//  
//                  //index 4 of matrix
//                  if ((int)bicubicImage.at<unsigned char>(j,i+1) < center)
//                                                  surroundingValues[4] = 0;
//                  else
//                                                  surroundingValues[4] = 1;
//  
//                  //index 5 of matrix
//                  if ((int)bicubicImage.at<unsigned char>(j+1,i-1) < center)
//                                                  surroundingValues[5] = 0;
//                  else
//                                                  surroundingValues[5] = 1;
//  
//                  //index 6 of matrix
//                  if ((int)bicubicImage.at<unsigned char>(j+1,i) < center)
//                                                  surroundingValues[6] = 0;
//                  else
//                                                  surroundingValues[6] = 1;
//  
//                  //index 7 of matrix
//                  if ((int)bicubicImage.at<unsigned char>(j+1,i+1) < center)
//                                                  surroundingValues[7] = 0;
//                  else
//                                                  surroundingValues[7] = 1;
//  
//                                          //setting the binary value to decimal value for the pixel
//                  LBPImage.at<unsigned char>(j,i) = (surroundingValues[0]*pow(2,7)+
//  				surroundingValues[1]*pow(2,6)+
//  				surroundingValues[2]*pow(2,5)+
//  				surroundingValues[3]*pow(2,4)+
//  				surroundingValues[4]*pow(2,3)+
//  				surroundingValues[5]*pow(2,2)+
//  				surroundingValues[6]*pow(2,1)+
//  				surroundingValues[7]*pow(2,0));
//    }
//  }
//  return LBPImage;
//}


        //return value: LBP matrix
    //input: bicubic frame
        cv::Mat LBP (cv::Mat bicubicImage){

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
                        LBPImage.at<unsigned char>(j,i) = (surroundingValues[0]*pow(2,7)+surroundingValues[1]*pow(2,6)+surroundingValues[2]*pow(2,5)
                                                                                                                        +surroundingValues[4]*pow(2,4)+surroundingValues[7]*pow(2,3)+surroundingValues[6]*pow(2,2)
                                                                                                                        +surroundingValues[5]*pow(2,1)+surroundingValues[3]*pow(2,0));
                                }
        }
                //STILL NEED TO IMPLEMENT THE EDGE CASES
                
                //--------------------------------------------------------------------------------------------------//
                center = (int)bicubicImage.at<unsigned char>(0,0);

                //top left corner
                if((int)bicubicImage.at<unsigned char>(0,1) < center)
                        surroundingValues[4] = 0;
                else
                        surroundingValues[4] = 1;

                if((int)bicubicImage.at<unsigned char>(1,0) < center)
                        surroundingValues[6] = 0;
                else
                        surroundingValues[6] = 1;

                if((int)bicubicImage.at<unsigned char>(1,1) < center)
                        surroundingValues[7] = 0;
                else
                        surroundingValues[7] = 1;

                LBPImage.at<unsigned char>(0,0) = (surroundingValues[4]*pow(2,4) +surroundingValues[6]*pow(2,2)
                        +surroundingValues[7]*pow(2,3));

                
                //top right corner
                center = (int)bicubicImage.at<unsigned char>(0,maxWidth-1);

                if((int)bicubicImage.at<unsigned char>(0,maxWidth-2) < center)
                        surroundingValues[3] = 0;
                else
                        surroundingValues[3] = 1;

                if((int)bicubicImage.at<unsigned char>(1,maxWidth-2) < center)
                        surroundingValues[5] = 0;
                else
                        surroundingValues[5] = 1;

                if((int)bicubicImage.at<unsigned char>(1,maxWidth-1) < center)
                        surroundingValues[6] = 0;
                else
                        surroundingValues[6] = 1;

                LBPImage.at<unsigned char>(0,maxWidth-1) = (surroundingValues[3]*pow(2,0) +surroundingValues[5]*pow(2,1)
                        +surroundingValues[6]*pow(2,2));
                
                //bottom left corner
                center = (int)bicubicImage.at<unsigned char>(maxHeight-1,0);

                if((int)bicubicImage.at<unsigned char>(maxHeight-2,0) < center)
                        surroundingValues[1] = 0;
                else
                        surroundingValues[1] = 1;

                if((int)bicubicImage.at<unsigned char>(maxHeight-2,1) < center)
                        surroundingValues[2] = 0;
                else
                        surroundingValues[2] = 1;

                if((int)bicubicImage.at<unsigned char>(maxHeight-1,1) < center)
                        surroundingValues[4] = 0;
                else
                        surroundingValues[4] = 1;

                LBPImage.at<unsigned char>(maxHeight-1,0) = (surroundingValues[1]*pow(2,6)+surroundingValues[2]*pow(2,5)
                +surroundingValues[4]*pow(2,4));

                
                //bottom right corner
                center = (int)bicubicImage.at<unsigned char>(maxHeight-1,maxWidth-1);

                if((int)bicubicImage.at<unsigned char>(maxHeight-2,maxWidth-2) < center)
                        surroundingValues[0] = 0;
                else
                        surroundingValues[0] = 1;

                if((int)bicubicImage.at<unsigned char>(maxHeight-2,maxWidth-1) < center)
                        surroundingValues[1] = 0;
                else
                        surroundingValues[1] = 1;

                if((int)bicubicImage.at<unsigned char>(maxHeight-1,maxWidth-2) < center)
                        surroundingValues[3] = 0;
                else
                        surroundingValues[3] = 1;

                LBPImage.at<unsigned char>(maxHeight-1,maxWidth-1) = (surroundingValues[0]*pow(2,7)+surroundingValues[1]*pow(2,6)
                +surroundingValues[3]*pow(2,0));

                
                //top row
                for (int i=1; i < maxWidth-1;i++)
                {
                        int j=0;

                        center = (int)bicubicImage.at<unsigned char>(j,i);

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
                        LBPImage.at<unsigned char>(j,i) =(surroundingValues[3]*pow(2,0)+surroundingValues[4]*pow(2,4)
                        +surroundingValues[5]*pow(2,1)+surroundingValues[6]*pow(2,2)+surroundingValues[7]*pow(2,3));
                }
                
                
                //left column
                for (int j=1; j < maxHeight-1;j++)
                {
                        int i=0;

                        center = (int)bicubicImage.at<unsigned char>(j,i);

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

                        //index 4 of matrix
                        if ((int)bicubicImage.at<unsigned char>(j,i+1) < center)
                                surroundingValues[4] = 0;
                        else
                                surroundingValues[4] = 1;

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
                        LBPImage.at<unsigned char>(j,i) = (surroundingValues[1]*pow(2,6)+surroundingValues[2]*pow(2,5)
                                +surroundingValues[4]*pow(2,4)+surroundingValues[6]*pow(2,2)+surroundingValues[7]*pow(2,3));
                }
        
                //right column
                for (int j=1; j < maxHeight-1;j++)
                {
                        int i=maxWidth-1;

                        center = (int)bicubicImage.at<unsigned char>(j,i);


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

                         //index 3 of matrix
                        if ((int)bicubicImage.at<unsigned char>(j,i-1) < center)
                                surroundingValues[3] = 0;
                        else
                                surroundingValues[3] = 1;

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

                        //setting the binary value to decimal value for the pixel
                        LBPImage.at<unsigned char>(j,i) = (surroundingValues[0]*pow(2,7)+surroundingValues[1]*pow(2,6)
                        +surroundingValues[3]*pow(2,0)+surroundingValues[5]*pow(2,1)+surroundingValues[6]*pow(2,2));
                }

                
                //bottom column
                for (int i=1; i < maxWidth-1;i++)
                {
                        int j=maxHeight-1;

                        center = (int)bicubicImage.at<unsigned char>(j,i);


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

                        //setting the binary value to decimal value for the pixel
                        LBPImage.at<unsigned char>(j,i) = (surroundingValues[0]*pow(2,7)+surroundingValues[1]*pow(2,6)
                        +surroundingValues[2]*pow(2,5)+surroundingValues[3]*pow(2,0)+surroundingValues[4]*pow(2,4));
                }
                
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
      cv::Mat coefficientMatrixOfHRLBP (cv::Mat HRLBP)
      {
                
        cv::Mat coefficentC = HRLBP.clone();        //making another copy of the input (bicubic) image (to be modified and returned)
        int maxWidth = coefficentC.rows;                        //find widith of the bicubic image
        int maxHeight = coefficentC.cols;                        //find height of the bicubic image
        int matrixValue;
	
                //rows
        for (int i=1; i < maxHeight; i++)
	{
                                //columns
          for (int j=1; j < maxWidth; j++)
          {
            matrixValue = (int)HRLBP.at<unsigned char>(j,i);
            
            std::bitset<8> b1(matrixValue);
            int value =0;
            for (int i = 0; i < 8; i++)
            {
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
          for (int i=1; i < maxHeight; i++)
	  {
                                  //columns
            for (int j=1; j < maxWidth; j++)
	    {
              output.at<unsigned char>(j,i) = (int)HRLBP.at<unsigned char>(j,i)*(int)ALD.at<unsigned char>(j,i) ;
            }
          }
                
          return output;
        }

//        cv::Mat reconstructedHRI (cv::Mat ALD, cv::Mat LRI, cv::Mat HRLBP ,cv::Mat bicubicUpSample, double scaleFactor)
//	{
//                
//          float lambda = 0.5;                
//          cv::Mat reconstructedHRI;
//          cv::Size outImageSize(0,0);        //dummy variable for resize
//          
//          //cv::Mat ALD = getALD(bicubicUpSample,15);
//
//          //cv::Mat CoefficentMatrix = coefficientMatrixOfHRLBP(HiResImage);
//          cv::Mat CoefficentMatrix = coefficientMatrixOfHRLBP(HRLBP);
//          cv::imshow ("Coefficent Matrix", CoefficentMatrix);
//
//          cv::Mat CD = elementWiseMultiply(CoefficentMatrix, ALD);
//          cv::Mat LCD = lambda*CD;
//
//          cv::Mat UY;
//          cv::resize(LRI,UY, outImageSize ,scaleFactor,scaleFactor,CV_INTER_NN);
//
//          cv::Mat HTUY;
//          HTUY = HRLBP.clone();
//          cv::blur(UY,HTUY,cv::Size(3,3));
//          //cv::GaussianBlur(UY, HTUY, cv::Size(5,5), 5);
//          //cv::addWeighted(UY, 1.5, HTUY, -0.5, 0, UY);
//
////                int maxWidth = HTUY.rows;                        //find widith of the bicubic image
////        int maxHeight = HTUY.cols;                        //find height of the bicubic image
////
////
////                int maxWidthALD = LCD.rows;                        //find widith of the bicubic image
////        int maxHeightALD = LCD.cols;                        //find height of the bicubic image
//
//                //cout << "HTUY: " << maxWidth <<        "                " << maxHeight << endl;
//                //cout << "LCD: "<< maxWidthALD << "                " << maxHeightALD << endl;
//
//
//          reconstructedHRI = HTUY + LCD;
//          return reconstructedHRI;
//          //return ALD;
//        }

cv::Mat reconstructedHRI (cv::Mat ALD, cv::Mat LRI, cv::Mat HRLBP ,cv::Mat bicubicUpSample, double scaleFactor)
        {
                
                double lambda = 0.1;                
                cv::Mat reconstructedHRI;
                cv::Size outImageSize(0,0);        //dummy variable for resize

                cv::Mat coeffMatrix = coefficientMatrixOfHRLBP(HRLBP);

               // cv::Mat ALD = getALD(bicubicUpSample,3);
                //cv::Mat CD = elementWiseMultiply(CoefficentMatrix, ALD);
                

                cv::Mat CD = elementWiseMultiply(coeffMatrix, ALD);
                
                cv::Mat LCD = lambda*CD;
        
                cv::Mat UY;
                cv::resize(LRI,UY, outImageSize ,scaleFactor,scaleFactor,CV_INTER_NN);
                

                cv::Mat HTUY;
                HTUY = HRLBP.clone();
                cv::blur(UY,HTUY,cv::Size(3,3));

                reconstructedHRI = HTUY + LCD;
                return reconstructedHRI;
                //return ALD;
        }


        // PARAM: original unscaled Y-channel of image to scale.
	// > bicubic upsample of the original Y-channel
	// > ALD of bicubic upsample of the original Y-channel.
	// > Amount that image is scaled by.
        cv::Mat getFinalTexture (cv::Mat originalImage, cv::Mat bicubicImage, cv::Mat ALD, double scaleFactor)
        { 
          cv::Mat LBPImage = LBP(originalImage);
          cv::Mat LBPBicubic = LBP(bicubicImage);
          cv::imshow("LBP", LBPImage);
          cv::imshow("Bicubic LBP", LBPBicubic);

          cv::Mat HiResImage = HRLBP (originalImage, bicubicImage, scaleFactor);
          cv::imshow ("HR LBP", HiResImage);

          cv::Mat newHRI;
          newHRI = reconstructedHRI(ALD,originalImage, HiResImage, bicubicImage, scaleFactor);
          cv::imshow ("reconstructed HRI", newHRI);

          //missing equation 14

          return newHRI;
        }

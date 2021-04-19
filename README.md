# fyp-backend
- [Webapp Frontend Link](https://fyp-frontend-app.herokuapp.com/) -> https://fyp-frontend-app.herokuapp.com/
- [Backend API Link](https://recommender-fyp.herokuapp.com)  ->   https://recommender-fyp.herokuapp.com

## Usage :

## For electronics category :
1. [GET getpopular] https://recommender-fyp.herokuapp.com/electronics/getpopular
2. [GET getpopular] https://recommender-fyp.herokuapp.com/electronics/getpopular
3. [GET getsearchimgdata] https://recommender-fyp.herokuapp.com/electronics/getsearchimgdata
4. [GET getdata] https://recommender-fyp.herokuapp.com/electronics/getdata
5. [POST getrecommendations]) https://recommender-fyp.herokuapp.com/electronics/getrecommendations
   ### request BODY in raw->json

                        { 
                            "all_products" : [	
                              {
                                "prodId":"1400532654",
                                "rating":5
                              }
                              ,

                              {
                                "prodId":"B001NJ0D0Y",
                                "rating":3
                              }
                            ]

                          }
                          
6. [POST getsimilarimage] https://recommender-fyp.herokuapp.com/electronics/getsimilarimage
   ### request BODY in raw->json
                       {
                          "prod_ID" : "B001NJ0D0Y"

                        }

7. [POST] https://recommender-fyp.herokuapp.com/electronics/getrecommendations_svd 
   ### request BODY in raw->json
                     {
                      "user_id" : "A5JLAU2ARJ0XX" ,
                      "all_products" : [
                        {
                          "prodId":"B00FJRS5BA",
                          "rating":5
                        }
                        ,

                      {
                          "prodId":"B000EVM5DK",
                          "rating":3
                        }
                      ]

                    }
                     
                     
## For sports category :
1. [GET getpopular] 'https://recommender-fyp.herokuapp.com/sports/getpopular'
2. [GET getsearchimgdata] 'https://recommender-fyp.herokuapp.com/sports/getsearchimgdata'
3. [GET getdata] 'https://recommender-fyp.herokuapp.com/sports/getdata'
4. [POST getrecommendations] 'https://recommender-fyp.herokuapp.com/sports/getrecommendations'
5. [POST getsimilarimage] 'https://recommender-fyp.herokuapp.com/sports/getsimilarimage'
                     
## For Cellphone category :
1. [GET getpopular] 'https://recommender-fyp.herokuapp.com/cellphone/getpopular'
2. [GET getsearchimgdata] 'https://recommender-fyp.herokuapp.com/cellphone/getsearchimgdata'
3. [GET getdata] 'https://recommender-fyp.herokuapp.com/cellphone/getdata'
4. [POST getrecommendations] 'https://recommender-fyp.herokuapp.com/cellphone/getrecommendations'
5. [POST getsimilarimage] 'https://recommender-fyp.herokuapp.com/cellphone/getsimilarimage'
                     

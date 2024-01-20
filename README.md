# sul-ijoa_DeepLearning

### 이미지가 음식인지 아닌지 예측 후 음식의 이름과 음식점 종류 추천
**EndPoint:** /api/image-analyze  
**Method:** POST  
**Content-Type:** multipart/form-data  

**참고 사항:**  
서버는 Flask를 기반으로 하며, 이미지 검증을 위해 Google Cloud Vision API를 활용합니다.  
Vision API의 응답에서
라벨의 가장 많은 %를 차지하는 것이 Food일 경우, 서비스는 HTTP 200으로 응답하고 그렇지 않으면 HTTP 500으로 응답합니다.

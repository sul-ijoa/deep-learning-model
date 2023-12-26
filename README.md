# sul-ijoa_DeepLearning

### 이미지를 form-data로 받아서 음식 이미지 검증 후, 어떤 음식인지 예측
**EndPoint:** /api/image-analyze  
**Method:** POST  
**Content-Type:** multipart/form-data  

**참고 사항:**  
서버는 Flask를 기반으로 하며, 이미지 검증을 위해 Google Cloud Vision API를 활용합니다.  
Vision API의 응답에서
라벨과 객체 정보 안에 Food가 존재하고 Dessert는 존재하지 않는 경우, 서비스는 HTTP 200으로 응답합니다. 그렇지 않으면 HTTP 500으로 응답합니다.  
이미지 분석 중에 오류가 발생한 경우, 자세한 내용을 포함한 오류 응답이 제공됩니다. 

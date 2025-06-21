# ocr-server

This projects creates a docker image which can process pictures of meters and extract the values.
Status: concept work

Currently it's using the following models:
- tesseract
- Digital (AI on the EDGE)
- Analog (AI onn the EDGE)

Send a request f.e. via curl:

curl -X POST http://localhost:5000/segment   -F "identifier=wasserzaehler"   -F "image=@./t4.jpg"

{"identifier":"wasserzaehler","results":[{"id":"a1","value":"00016.0"},{"id":"a2","value":9.127448058121708},{"id":"a3","value":3.881748198574262},{"id":"anzeige1","value":1.2311287412814798},{"id":"a5","value":3.345734090629815}]}

You can use http://127.0.0.1:5000/ to upload a new meter picture and to generate and test the yaml config for this meter.


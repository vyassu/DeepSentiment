if [ ! -d "./Data" ]
then
	mkdir Data
fi
if [ ! -d "./inputdata" ]
then
	mkdir inputdata
fi

cp -R ../../WebInterface/* .

python ./webservice.py
echo "Webservice started!! Use URL http://localhost:5000/deepsentiment"

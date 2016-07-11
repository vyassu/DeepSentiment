navigator.getUserMedia  = navigator.getUserMedia ||
                          navigator.webkitGetUserMedia ||
                          navigator.mozGetUserMedia ||
                          navigator.msGetUserMedia;

window.AudioContext = window.AudioContext ||
                      window.webkitAudioContext;
var context = new AudioContext();
var analyser = context.createAnalyser();
var oncounter = false;
var array;

var canvas = document.querySelector('.visualization');
var canvasCtx = canvas.getContext("2d");

WIDTH = canvas.width;
HEIGHT = canvas.height;

canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);


function errorCallback(data)
{
	console.log(data);
}

function recordSound()
{	
	console.log("in control");
	if (oncounter==false)
	{
		context.resume();
		navigator.getUserMedia({audio: true}, function(stream) {
			var source = context.createMediaStreamSource(stream);     
		    console.log(stream);             
	  		source.connect(analyser);  
	  		console.log(analyser)   ; 
	  		analyser.connect(context.destination);
	  		array = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(array);
            console.log(array);
            callwebservice(array);

        drawVisual = requestAnimationFrame(recordSound);
        canvasCtx.fillStyle = 'rgb(200, 200, 200)';
    	canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

    	canvasCtx.lineWidth = 2;
    	canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

    	canvasCtx.beginPath();

    	var sliceWidth = WIDTH * 1.0 / array.length;
    	var x = 0;

    	for(var i = 0; i < array.length; i++) {
   			var v = array[i] / 255.0;
        	var y = v * HEIGHT/2;

        	if(i === 0) {
          	canvasCtx.moveTo(x, y);
        	} else {
          	canvasCtx.lineTo(x, y);
        	}
			x += sliceWidth;
      	}

     	canvasCtx.lineTo(canvas.width, canvas.height/2);
      	canvasCtx.stroke();
		  }, errorCallback);
		oncounter=true;
	}
	else
	{
		context.suspend();
		oncounter=false;
	}
}

function callwebservice(var_data)
{
	var soundData="[";
	for(var i=0; i < var_data.length-1;i++)
		{
			soundData = soundData+var_data[i]+",";
			//console.log(var_data[i]);
		}

	var webData = {
					"data" :soundData = soundData+var_data[var_data.length-1]+"]",
					"samplerate": context.sampleRate
				  }

	$.ajax({
            type: "POST",
            method: "POST",
            url: "http://localhost:5000/deepsentimentweb",
            contentType: "application/json",
            datatype:"text",
            data: JSON.stringify(webData),
            timeout : 5000,
            success : function(data)
            {
            	console.log(data);
            },
            error: function(data)
            {
            	
            	console.log(data);
            }
        });
}





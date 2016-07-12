navigator.getUserMedia  = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia ||
                          navigator.msGetUserMedia;

window.AudioContext = window.AudioContext || window.webkitAudioContext;
var context = new AudioContext();
var analyser = context.createAnalyser();
var oncounter = false;
var microphone_output_buffer;
var i=1;

var canvas = document.querySelector('.visualization');
var canvasCtx = canvas.getContext("2d");
WIDTH = canvas.width;
HEIGHT = canvas.height;
canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);


function errorCallback(data)
{
	console.log(data);
}

function process_microphone_buffer(event) {

        
		if(oncounter==true)
		{
			if (i==1)
			{
				microphone_output_buffer = new Float32Array(new ArrayBuffer(4096));
				microphone_output_buffer.set(event.inputBuffer.getChannelData(0),0);
			}
			else
			{
				var temp_buffer = new Float32Array(new ArrayBuffer(4096*(i+1)));
				temp_buffer.set(microphone_output_buffer,0);
				temp_buffer.set(event.inputBuffer.getChannelData(0),microphone_output_buffer.length);
				microphone_output_buffer = temp_buffer;
				temp_buffer = null;
			}
			i++;
			
		}
		else
		{
			microphone_output_buffer = new Float32Array(new ArrayBuffer(4096));
			i=1;
		}
}

function processTimeData(stream)
{
	gain_node = context.createGain();
    gain_node.connect( context.destination );

    microphone_stream = context.createMediaStreamSource(stream);
    microphone_stream.connect(gain_node); 

    script_processor_node = context.createScriptProcessor(1024, 1, 1);
    script_processor_node.onaudioprocess = process_microphone_buffer;

    microphone_stream.connect(script_processor_node);
}


	/*
	//callwebservice(array);

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
		  */




function recordSound()
{	
	if (oncounter==false)
		{
			context.resume();
			navigator.getUserMedia({audio: true}, function(stream) {
				processTimeData(stream);
				oncounter = true;
			}, errorCallback);
		}
		else
		{
			context.suspend();
			callwebservice(false);
			i=1;
			oncounter = false;
			microphone_output_buffer = new Float32Array(new ArrayBuffer(4096));
		}
}

var filename;
$("#filedata").on("change", function(event){
	filename = event.target.files;
}
);

function sendData()
{
	var filedata = new FileReader();
    $.each(filename, function(index,file)
    {
    	filedata.readAsDataURL(file);
    	filedata.onloadend = function(event)
    	{
    		console.log(filedata.result);
    		microphone_output_buffer = filedata.result;
    		callwebservice(true);
    	}
    });
}


function fillText(data)
{

}

function callwebservice(fileupload)
{
	var webdata,URL;
	if(fileupload==false)
	{
		var soundData="["+microphone_output_buffer.join();
		soundData = soundData+"]";
		webData = {
						"data" :soundData,
						"samplerate": context.sampleRate
					  }
		webdata =  JSON.stringify(webData);
		URL= "http://localhost:5000/deepsentimentweb"
	}
	else
	{
		webdata = microphone_output_buffer;
		URL = "http://localhost:5000/deepsentifile";
	}

	$.ajax({
            type: "POST",
            method: "POST",
            url: URL,
            contentType: "application/json",
            datatype:"text",
            data: webdata,
            timeout : 5000,
            success : function(data)
            {
            	canvasCtx.font = "24px Times New Roman";
            	canvasCtx.fillStyle = "white";
				canvasCtx.fillText("What you spoke:: "+data.input,10,50);
				canvasCtx.fillText(data.result,10,90);
            },
            error: function(data)
            {
            	
            	alert("Backend server not responding!! Please try again after sometime");
            	console.log(data);
            }
        });
}





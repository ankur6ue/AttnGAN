/**
 * Client side of PyTorch Detection Web API
 * Initial version taken from webrtcHacks - https://webrtchacks.com
 */
const s = document.getElementById('attnGAN');
 const hostType = s.getAttribute("data-apiServer")
if (hostType == 'localhost')
	apiServer = "http://127.0.0.1:5000" // must be just like this. using 0.0.0.0 for the IP doesn't work! 
else
	apiServer = "http://52.90.180.232/attn_gan_impl"

captions = []
var activeModel = ""

captions['bird'] = 
[
	"this bird is red with white and has a very short beak",
	"the bird has a yellow crown and a black eyering that is round",
	"this bird has a green crown black primaries and a white belly",
	"this bird has wings that are black and has a white belly",
	"this bird has wings that are red and has a yellow belly",
	"this bird has wings that are blue and has a red belly",
	"this is a small light gray bird with a small head and green crown, nape and some green coloring on its wings",
	"his bird is black with white and has a very short beak",
	"this small bird has a deep blue crown, back and rump and a bright white belly",
	"this is a blue bird with a white throat, breast, belly and abdomen and a small black pointed beak",
	"yellow abdomen with a black eye without an eye ring and beak quite short",
	"this bird is yellow with black on its head and has a very short beak",
	"this bird has wings that are black and has a white belly",
	"this bird has a white belly and breast with a short pointy bill",
	"this bird is red and white in color with a stubby beak and red eye rings",
	"a small red and white bird with a small curved beak"
]

captions['coco'] = 
[
	"a photo of a homemade swirly pasta with broccoli carrots and onions",
	"a fruit stand display with bananas and kiwi",
	"the girl is surfing a small wave in the water",
	"a large red and white boat floating on top of a lake",
	"a herd of cows that are grazing on the grass",
	"a herd of sheep grazing on a lush green filed",
	"an old clock next to a light post in front of a steeple",
	"an image of a girl eating a large slice of pizza",
	"a clock that is on the side of a tower",
	"well lit skyscrapers and a clock tower in the evening",
	"a man kite boarding in the ocean next to a sandy beach",
	"flat screen television on top of an old tv console"
]

errorCodes = 
["bad caption"]

function LoadModel(modelType)
{

	PopulateExampleCaptions(modelType)
	
	let xhr = new XMLHttpRequest();
	var method = apiServer + '/init/' + modelType
	xhr.open('GET', method, true);
	xhr.onload = function () {
		if (this.status === 200) {
			$('#status').val(this.response);
			$('.activeModel').val("Active Model: " + modelType)
		}
		else
		{
			$('#status').val("Problem Initializing GAN Model");
		}
	};
	xhr.send()
}

function PopulateExampleCaptions(modelType)
{
	var dropdown = $(".dropdown-menu.example-captions");
	$(".dropdown-menu.example-captions").empty();
	for( var i = 0; i < captions[modelType].length; i++ )
	{ 
		var o = captions[modelType][i];
		dropdown.append("<li class=\"ExampleCaptionDropDownListItem\" data-name=" + "\"" + o + "\"" + "role=\"presentation\"><a role=\"menuitem\" tabindex=\"-1\" href=\"#\">" +  o + "</a></li>") 
	}      
	$('.ExampleCaptionDropDownListItem').click(function(e) {
		var name = e.currentTarget;
		var caption = name.getAttribute("data-name")
		console.log(caption);
		$('#caption').val(caption);
	});
	
}
document.addEventListener("DOMContentLoaded", function() {
	$('.ModelDropDownListItem').click(function(e) {
		var name = e.currentTarget;
		var modelName = name.getAttribute("data-name")
		console.log(modelName);
		LoadModel(modelName)
		activeModel = modelName;
	});
	
	}, false);

function onSubmitCaption()
{
	if (!$.trim($("#caption").val())) {// textarea is empty or contains only white-space
		$('#status').val("Please enter a caption");
		return;
	}
    
	// Clear status
	$('#status').val("");
	
	let xhr = new XMLHttpRequest();
	var method = apiServer + '/generate/' + $('#caption').val() + '/' + activeModel;
    xhr.open('POST', method, true);
	xhr.onload = function () {
        if (this.status === 200) {
			if (errorCodes.includes(this.response)){
				// There is some error
				$('#status').val(this.response);
			}
			else
			{
				var ims = JSON.parse(this.response)
				var im_256 = ims[0]['im_256']
				var im_128 = ims[0]['im_128']
				var im_64 = ims[0]['im_64']
				var fp_time = ims[0]['fp_time']
				
				$('#GeneratedImages_256').append('<img src="data:image/png;base64,' + im_256 + '" class="img-rounded"/>');
				$('#GeneratedImages_128').append('<img src="data:image/png;base64,' + im_128 + '" class="img-rounded"/>');
				$('#GeneratedImages_64').append('<img src="data:image/png;base64,' + im_64 + '" class="img-rounded"/>');
				
				$('#status').val("Image generated in " + parseFloat(fp_time).toFixed(2) + " seconds");
			}
		}
		if (this.status === 500) { // Internal Server Error
			// There is some error
			$('#status').val("Internal Server Error:" + this.response);
			
		}
	};
	xhr.timeout = 4000;
	xhr.send(null)
}


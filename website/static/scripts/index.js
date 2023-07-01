import WaveSurfer from 'https://unpkg.com/wavesurfer.js@beta'

const STARTING_VOLUME = 0.5;
var wavesurfer = null;

// Execute code when the page finishes loading
window.addEventListener('DOMContentLoaded', function() {
	var dropArea = document.getElementById('drop-area');
	// Prevent default behavior (Prevent file from being opened)
	dropArea.addEventListener('dragover', function (e) {
		e.preventDefault();
	});
	
	// Handle dropped files
	dropArea.addEventListener('drop', function (e) {
		e.preventDefault();
		
		var files = e.dataTransfer.files;
		if (files.length > 0) {
			var soundFile = files[0]; // Only grab the first file

			// Set the dropped sound file as the audio source
			var soundFileURL = URL.createObjectURL(soundFile);
			wavesurfer = WaveSurfer.create({
				autplay: 'false',
				container: '#waveform',
				waveColor: '#88c0d0',
				progressColor: '#7cb0be'
			});

			wavesurfer.load(soundFileURL);
			wavesurfer.on('interaction', () => {
				wavesurfer.play()
				document.getElementById('playToggleButton').textContent = 'Pause';
			})

			setupPlayButton();
			setupVolumeSlider();
		}
	});
});

function setupPlayButton() {
	var toggleButton = document.getElementById('playToggleButton');
	toggleButton.addEventListener('click', function() {
	  if (wavesurfer.isPlaying()) {
		wavesurfer.pause();
		toggleButton.textContent = 'Play';
	  } else {
		wavesurfer.play();
		toggleButton.textContent = 'Pause';
	  }
	});
}

function setupVolumeSlider() {
	var volumeSlider = document.querySelector('#volumeSlider');
	volumeSlider.value = STARTING_VOLUME;
	wavesurfer.setVolume(STARTING_VOLUME);

	volumeSlider.addEventListener('input', function(event) {
		wavesurfer.setVolume(event.target.value);
	});
}
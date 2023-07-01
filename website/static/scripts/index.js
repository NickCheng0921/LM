import WaveSurfer from 'https://unpkg.com/wavesurfer.js@beta'
import Spectrogram from 'https://unpkg.com/wavesurfer.js@beta/dist/plugins/spectrogram.js'
import { velocity_blue_map } from "../scripts/colorMap.js"

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
		document.getElementById('drop-area').style.display = 'none';
		
		var files = e.dataTransfer.files;
		if (files.length > 0) {
			var soundFile = files[0]; // Only grab the first file

			// Set the dropped sound file as the audio source
			var soundFileURL = URL.createObjectURL(soundFile);
			wavesurfer = WaveSurfer.create({
				autplay: 'false',
				container: '#waveform',
				waveColor: '#88c0d0',
				progressColor: '#7cb0be',
				url: soundFileURL,
			});

			wavesurfer.on('interaction', () => {
				wavesurfer.play()
			})

			setupPlayButton();
			setupVolumeSlider();
			setupSpectrogram();
		}
	});
});

function setupPlayButton() {
	var toggleButton = document.getElementById('playToggleButton');
	toggleButton.addEventListener('click', function() {
	  if (wavesurfer.isPlaying()) {
		wavesurfer.pause();
	  } else {
		wavesurfer.play();
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

function setupSpectrogram() {
	wavesurfer.registerPlugin(
		Spectrogram.create({
			labels: true,
			container: '#spectrogram',
			colorMap: velocity_blue_map,
		}),
	)
}
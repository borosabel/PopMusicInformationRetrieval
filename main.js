import WaveSurfer from "wavesurfer";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";

// Initialize the Regions plugin
const regions = RegionsPlugin.create();

// Create a WaveSurfer instance
const ws = WaveSurfer.create({
  container: "#waveform",
  waveColor: "rgb(200, 0, 200)",
  progressColor: "rgb(100, 0, 100)",
  url: "./test_data_wav/Waka Flocka Flame - Luv Dem Gun Sounds [Official Video].wav", // Make sure this path is correct
  plugins: [regions],
});

// Function for generating random color
const random = (min, max) => Math.random() * (max - min) + min;
const randomColor = () =>
  `rgba(${random(0, 255)}, ${random(0, 255)}, ${random(0, 255)}, 0.5)`;

// Create some regions
ws.on("decode", () => {
  regions.addRegion({
    start: 0,
    end: 8,
    content: "Resize me",
    color: randomColor(),
    drag: false,
    resize: true,
  });
  regions.addRegion({
    start: 9,
    end: 10,
    content: "Cramped region",
    color: randomColor(),
    minLength: 1,
    maxLength: 10,
  });
  regions.addRegion({
    start: 12,
    end: 17,
    content: "Drag me",
    color: randomColor(),
    resize: false,
  });

  regions.addRegion({
    start: 19,
    content: "Marker",
    color: randomColor(),
  });
  regions.addRegion({
    start: 20,
    content: "Second marker",
    color: randomColor(),
  });
});

// Enable drag selection
regions.enableDragSelection({
  color: "rgba(255, 0, 0, 0.1)",
});

regions.on("region-updated", (region) => {
  console.log("Updated region", region);
});

// Loop region functionality
let loop = true;
document.querySelector('input[type="checkbox"]').onclick = (e) => {
  loop = e.target.checked;
};

let activeRegion = null;
regions.on("region-in", (region) => {
  console.log("region-in", region);
  activeRegion = region;
});
regions.on("region-out", (region) => {
  if (activeRegion === region && loop) {
    region.play();
  } else {
    activeRegion = null;
  }
});
regions.on("region-clicked", (region, e) => {
  e.stopPropagation();
  activeRegion = region;
  region.play();
  region.setOptions({ color: randomColor() });
});

ws.on("interaction", () => {
  activeRegion = null;
});

// Zoom functionality
ws.once("decode", () => {
  document.querySelector('input[type="range"]').oninput = (e) => {
    const minPxPerSec = Number(e.target.value);
    ws.zoom(minPxPerSec);
  };
});

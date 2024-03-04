const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs-node");

// Load captcha images and labels
const captchaFolder = "a";
const captchaFiles = fs.readdirSync(captchaFolder);

const captchaData = captchaFiles.map((file) => {
  const label = path.parse(file).name; // Assuming filenames are the labels
  const imagePath = path.join(captchaFolder, file);
  const imageBuffer = fs.readFileSync(imagePath);
  return { imageBuffer, label };
});

const IMAGE_HEIGHT = 80;
const IMAGE_WIDTH = 240;
const BATCH_SIZE = 32;
const EPOCHS = 20;

// Preprocess and prepare data
const captchaTensors = captchaData.map(({ imageBuffer, label }) => {
  const image = tf.node.decodeImage(imageBuffer);
  const processedImage = image.resizeBilinear([IMAGE_HEIGHT, IMAGE_WIDTH]).toFloat().div(tf.scalar(255));

  // Convert to grayscale by taking the mean of RGB channels
  const grayscaleImage = processedImage.mean(2).expandDims(2);

  return { image: grayscaleImage, label };
});

const imageTensors = tf.stack(captchaTensors.map(({ image }) => image));
const labelTensors = tf.tensor2d(
  captchaTensors.map(({ label }) => label.split("").map(Number)),
  [captchaTensors.length, 6]
);

// Define the model
const model = tf.sequential({
  layers: [
    tf.layers.conv2d({
      inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, 1], // Changed inputShape to [80, 240, 1]
      kernelSize: 3,
      filters: 32,
      activation: "relu",
    }),
    tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
    tf.layers.flatten(),
    tf.layers.dense({ units: 128, activation: "relu" }),
    tf.layers.dense({ units: 6, activation: "softmax" }),
  ],
});

model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

// Train the model
model.fit(imageTensors, labelTensors, {
  batchSize: BATCH_SIZE,
  epochs: EPOCHS,
  validationSplit: 0.2,
  callbacks: tf.node.tensorBoard("/logs"),
}).then((history) => {
  console.log(history.history);
  // Save the model
  model.save("file://./saved_model").then(() => {
    console.log("Model saved.");
  });
});

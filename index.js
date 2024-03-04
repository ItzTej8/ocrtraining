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
const BATCH_SIZE = 1;
const EPOCHS = 10;

const captchaTensors = captchaData.map(({ imageBuffer, label }) => {
  const image = tf.node.decodeImage(imageBuffer);

  // Ensure that images have three channels (remove alpha channel if present)
  const processedImage = image.slice([0, 0, 0], [-1, -1, 3]);

  const resizedImage = tf.image.resizeBilinear(processedImage.expandDims(0), [
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
  ]);

  const normalizedImage = resizedImage.div(tf.scalar(255));
  return { image: normalizedImage, label };
});

const imageTensors = tf.concat(captchaTensors.map(({ image }) => image));

// Extract labels from file names
const labelTensors = tf.tensor2d(
  captchaTensors.map(({ label }) => label.split("").map(Number)),
  [captchaTensors.length, 6],
);

console.log(labelTensors.shape);

if (imageTensors.shape[0] !== labelTensors.shape[0]) {
  throw new Error("Mismatch in the number of labels and images");
}

/////////////////////////////////////////////// Define the model
const model = tf.sequential({
  layers: [
    tf.layers.conv2d({
      inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
      kernelSize: 3,
      filters: 16,
      activation: "relu",
    }),
    tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
    tf.layers.flatten(),
    tf.layers.dense({ units: 128, activation: "relu" }),
    // Change units to match the number of classes (6 in this case)
    tf.layers.dense({ units: 6, activation: "softmax" }),
  ],
});

model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

// Train the model
model
  .fit(imageTensors, labelTensors, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS,
    shuffle: true,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}: Loss - ${logs.loss.toFixed(
            4,
          )}, Accuracy - ${logs.acc.toFixed(4)}`,
        );
      },
    },
  })
  .then(() => {
    // Save the model
    model.save("file://./saved_model").then(() => {
      console.log("Model saved.");
    });
  });

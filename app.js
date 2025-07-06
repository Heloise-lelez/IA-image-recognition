const CLASSES = ["Lynx", "Salamandre", "Tortue"];
const inputFile = document.getElementById("file");
const preview = document.getElementById("preview");
const resultDiv = document.getElementById("result");
const uploadForm = document.getElementById("upload-form");
const runModelButton = document.getElementById("runModel");

let session = null;

// Cacher le bouton tant que le modèle n'est pas chargé
runModelButton.style.display = "none";

// Afficher l'aperçu de l'image sélectionnée
inputFile.addEventListener("change", () => {
  const file = inputFile.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function () {
      preview.src = reader.result;
      preview.style.display = "block";
    };
    reader.readAsDataURL(file);
  } else {
    preview.style.display = "none";
    preview.src = "";
  }
});

async function loadModel() {
  try {
    session = await ort.InferenceSession.create("my_model.onnx");
    console.log("✅ Modèle chargé avec succès");
    runModelButton.style.display = "inline-block"; // Affiche le bouton
  } catch (error) {
    console.error("❌ Erreur lors du chargement du modèle :", error);
  }
}

loadModel();

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (inputFile.files.length === 0) {
    alert("Choisis une image !");
    return;
  }

  const file = inputFile.files[0];
  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();

  try {
    const inputTensor = await preprocessImage(img);
    const feeds = { [session.inputNames[0]]: inputTensor };
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];

    // Softmax
    const expScores = output.data.map(Math.exp);
    const sumExpScores = expScores.reduce((a, b) => a + b, 0);
    const probabilities = expScores.map((score) => score / sumExpScores);

    let maxIndex = 0;
    let maxVal = probabilities[0];
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxVal) {
        maxVal = probabilities[i];
        maxIndex = i;
      }
    }

    let resultText = "";
    if (maxVal < 0.7) {
      resultText +=
        "Probablement ni un Lynx, ni une Salamandre, ni une Tortue <br>";
    } else {
      resultText += `Prédiction : <strong>${CLASSES[maxIndex]}</strong> (${(
        maxVal * 100
      ).toFixed(2)}%)<br>`;
    }

    resultDiv.innerHTML = resultText;
  } catch (error) {
    console.error("Erreur lors de l'inférence :", error);
  }
});

async function preprocessImage(imageElement) {
  const size = 128;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = size;
  canvas.height = size;
  ctx.drawImage(imageElement, 0, 0, size, size);

  const imageData = ctx.getImageData(0, 0, size, size);
  const data = imageData.data;
  const float32Data = new Float32Array(size * size * 3);

  for (let i = 0; i < size * size; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;

    float32Data[i] = (r - 0.485) / 0.229;
    float32Data[i + size * size] = (g - 0.456) / 0.224;
    float32Data[i + 2 * size * size] = (b - 0.406) / 0.225;
  }

  return new ort.Tensor("float32", float32Data, [1, 3, size, size]);
}

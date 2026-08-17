/* ============================================================
   AI DATA CLASSIFICATION - FRONTEND JAVASCRIPT
============================================================ */


/* ============================================================
   SAMPLE DATA
============================================================ */

const samples = {

    setosa: {
        sepal_length: 5.1,
        sepal_width: 3.5,
        petal_length: 1.4,
        petal_width: 0.2
    },

    versicolor: {
        sepal_length: 6.0,
        sepal_width: 2.9,
        petal_length: 4.5,
        petal_width: 1.5
    },

    virginica: {
        sepal_length: 6.3,
        sepal_width: 3.3,
        petal_length: 6.0,
        petal_width: 2.5
    }

};


/* ============================================================
   LOAD SAMPLE
============================================================ */

function loadSample(type) {

    const sample = samples[type];

    if (!sample) {
        return;
    }

    document.getElementById("sepal_length").value =
        sample.sepal_length;

    document.getElementById("sepal_width").value =
        sample.sepal_width;

    document.getElementById("petal_length").value =
        sample.petal_length;

    document.getElementById("petal_width").value =
        sample.petal_width;

}


/* ============================================================
   FORM SUBMISSION
============================================================ */

const form = document.getElementById("predictionForm");

if (form) {

    form.addEventListener("submit", async function (event) {

        event.preventDefault();

        const button = document.getElementById("predictBtn");

        button.disabled = true;

        button.innerHTML = "⏳ Classifying...";


        const data = {

            sepal_length:
                document.getElementById("sepal_length").value,

            sepal_width:
                document.getElementById("sepal_width").value,

            petal_length:
                document.getElementById("petal_length").value,

            petal_width:
                document.getElementById("petal_width").value

        };


        try {

            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)

            });


            const result = await response.json();


            if (!response.ok || !result.success) {

                showError(
                    result.error ||
                    "Unable to make prediction."
                );

                return;
            }


            showPrediction(result);


        } catch (error) {

            showError(
                "Could not connect to the Flask server."
            );

        } finally {

            button.disabled = false;

            button.innerHTML =
                "<span>🔮</span> Predict Flower";

        }

    });

}


/* ============================================================
   SHOW PREDICTION
============================================================ */

function showPrediction(result) {

    const placeholder =
        document.getElementById("resultPlaceholder");

    const content =
        document.getElementById("resultContent");

    if (!placeholder || !content) {
        return;
    }


    placeholder.classList.add("hidden");

    content.classList.remove("hidden");


    const flowerName =
        result.prediction.toLowerCase();


    const flowerEmojis = {

        setosa: "🌸",

        versicolor: "🌼",

        virginica: "🌺"

    };


    document.getElementById("resultFlower").textContent =
        flowerEmojis[flowerName] || "🌸";


    document.getElementById("predictionName").textContent =
        "Iris " +
        flowerName.charAt(0).toUpperCase() +
        flowerName.slice(1);


    let confidence =
        parseFloat(result.confidence);


    if (confidence <= 1) {
        confidence = confidence * 100;
    }


    confidence = Math.min(
        100,
        Math.max(0, confidence)
    );


    document.getElementById("confidenceValue").textContent =
        confidence.toFixed(2) + "%";


    setTimeout(function () {

        document.getElementById("confidenceBar").style.width =
            confidence + "%";

    }, 100);


    document.getElementById("predictionMessage").textContent =
        "The Decision Tree model classified this flower as " +
        flowerName +
        " with " +
        confidence.toFixed(2) +
        "% confidence.";

}


/* ============================================================
   ERROR MESSAGE
============================================================ */

function showError(message) {

    const placeholder =
        document.getElementById("resultPlaceholder");

    const content =
        document.getElementById("resultContent");


    if (!placeholder || !content) {
        alert(message);
        return;
    }


    content.classList.add("hidden");

    placeholder.classList.remove("hidden");


    placeholder.innerHTML = `

        <div class="result-icon">
            ⚠️
        </div>

        <h3>Prediction Error</h3>

        <p>${message}</p>

    `;

}
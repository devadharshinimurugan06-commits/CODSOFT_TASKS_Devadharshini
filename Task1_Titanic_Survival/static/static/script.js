
// ---------------------------------------
// EDIT PASSENGER SCENARIO
// ---------------------------------------

function editScenario() {
    const form = document.getElementById("predictionForm");

    if (form) {
        form.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

        const ageInput = document.getElementById("age");

        if (ageInput) {
            setTimeout(() => {
                ageInput.focus();
            }, 500);
        }
    }
}


// ---------------------------------------
// SURVIVAL GAUGE ANIMATION
// ---------------------------------------

function animateSurvivalGauge() {
    const gauges = document.querySelectorAll(".survival-gauge");

    gauges.forEach((gauge) => {
        const targetValue = gauge.style.getPropertyValue("--gauge-value");

        gauge.style.setProperty("--gauge-value", "0%");

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                gauge.style.setProperty("--gauge-value", targetValue);
            });
        });
    });
}


// ---------------------------------------
// AUTO SCROLL TO RESULTS
// ---------------------------------------

document.addEventListener("DOMContentLoaded", function () {

    const results = document.getElementById("results");
    const comparisonResults =
        document.getElementById("comparisonResults");

    // Scroll to normal prediction result
    if (results) {
        results.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

    // Scroll to comparison result
    else if (comparisonResults) {
        comparisonResults.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

    // Animate circular survival gauge
    animateSurvivalGauge();

    console.log("Titanic Survival Intelligence is ready.");

});

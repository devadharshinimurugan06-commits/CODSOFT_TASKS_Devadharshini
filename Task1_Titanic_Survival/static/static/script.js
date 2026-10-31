
function editScenario() {
    const form = document.getElementById("predictionForm");

    if (form) {
        form.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

        const ageInput = document.getElementById("age");
        if (ageInput) {
            ageInput.focus();
        }
    }
}

document.addEventListener("DOMContentLoaded", function () {
    const results = document.getElementById("results");

    if (results) {
        results.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

    console.log("Titanic Survival Intelligence is ready.");
});

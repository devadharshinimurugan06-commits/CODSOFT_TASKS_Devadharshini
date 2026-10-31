// =====================================
// TITANIC SURVIVAL INTELLIGENCE
// FRONTEND INTERACTIONS
// =====================================


// ABOUT DROPDOWN
const aboutDropdown = document.querySelector(".about-dropdown");
const aboutButton = document.querySelector(".about-button");

if (aboutButton && aboutDropdown) {
    aboutButton.addEventListener("click", function (event) {
        event.stopPropagation();

        const isOpen = aboutDropdown.classList.toggle("open");
        aboutButton.setAttribute("aria-expanded", isOpen);
    });

    document.addEventListener("click", function (event) {
        if (!aboutDropdown.contains(event.target)) {
            aboutDropdown.classList.remove("open");
            aboutButton.setAttribute("aria-expanded", "false");
        }
    });
}


// MOBILE NAVIGATION
const menuToggle = document.getElementById("menuToggle");
const navLinks = document.querySelector(".nav-links");

if (menuToggle && navLinks) {
    menuToggle.addEventListener("click", function () {
        navLinks.classList.toggle("open");
    });

    navLinks.querySelectorAll("a").forEach(function (link) {
        link.addEventListener("click", function () {
            navLinks.classList.remove("open");
        });
    });
}


// SEARCH THE APP
const searchInput = document.getElementById("appSearch");
const searchResults = document.getElementById("searchResults");

const appInformation = [
    {
        title: "Passenger Details",
        description: "Enter age, gender, passenger class, fare, family members and port of embarkation.",
        keywords: "passenger age gender sex class fare siblings spouse parents children embarkation",
        target: "#passenger"
    },
    {
        title: "Survival Prediction",
        description: "Submit passenger information to calculate the survival probability using the trained CatBoost model.",
        keywords: "prediction predict survival probability calculate model catboost result",
        target: "#passenger"
    },
    {
        title: "What-if Analysis",
        description: "Compare two hypothetical passenger profiles and their estimated survival probabilities.",
        keywords: "what if compare comparison scenario passenger profiles",
        target: "#comparison"
    },
    {
        title: "SHAP Explanation",
        description: "Understand the top three features that influenced the model prediction.",
        keywords: "shap explainable ai feature importance explanation influence",
        target: "#results"
    },
    {
        title: "About the Project",
        description: "Learn about the Titanic Survival Intelligence project and its machine learning technology.",
        keywords: "about project titanic history information",
        target: "#about"
    },
    {
        title: "Technology Used",
        description: "This application uses Python, Flask, CatBoost and SHAP.",
        keywords: "technology python flask catboost shap machine learning",
        target: "#technology"
    }
];

if (searchInput && searchResults) {
    searchInput.addEventListener("input", function () {
        const query = searchInput.value.trim().toLowerCase();

        searchResults.innerHTML = "";

        if (!query) {
            searchResults.hidden = true;
            return;
        }

        const matches = appInformation.filter(function (item) {
            const searchableText =
                item.title + " " +
                item.description + " " +
                item.keywords;

            return searchableText.toLowerCase().includes(query);
        });

        searchResults.hidden = false;

        if (matches.length === 0) {
            searchResults.innerHTML =
                '<div class="search-empty">No matching information found. Try searching for passenger, prediction, SHAP or comparison.</div>';
            return;
        }

        matches.forEach(function (item) {
            const resultItem = document.createElement("a");

            resultItem.href = item.target;
            resultItem.className = "search-result-item";

            const title = document.createElement("strong");
            title.textContent = item.title;

            const description = document.createElement("span");
            description.textContent = item.description;

            resultItem.appendChild(title);
            resultItem.appendChild(description);

            resultItem.addEventListener("click", function () {
                searchResults.hidden = true;
                searchInput.value = "";
            });

            searchResults.appendChild(resultItem);
        });
    });

    document.addEventListener("click", function (event) {
        if (!event.target.closest(".search-section")) {
            searchResults.hidden = true;
        }
    });

    searchInput.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
            searchResults.hidden = true;
            searchInput.value = "";
        }
    });
}


// EXPLORE ANOTHER PASSENGER
function editScenario() {
    const passengerSection = document.getElementById("passenger");
    const ageInput = document.getElementById("age");

    if (passengerSection) {
        passengerSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

    if (ageInput) {
        setTimeout(function () {
            ageInput.focus();
        }, 600);
    }
}


// ANIMATE SURVIVAL GAUGE
function animateSurvivalGauge() {
    const gauges = document.querySelectorAll(".survival-gauge");

    gauges.forEach(function (gauge) {
        const targetValue =
            gauge.style.getPropertyValue("--gauge-value");

        gauge.style.setProperty("--gauge-value", "0%");

        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                gauge.style.setProperty(
                    "--gauge-value",
                    targetValue
                );
            });
        });
    });
}


// ANIMATE COMPARISON BARS
function animateComparisonBars() {
    const bars = document.querySelectorAll(".progress-fill");

    bars.forEach(function (bar) {
        const targetWidth = bar.style.width;

        bar.style.width = "0%";

        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                bar.style.width = targetWidth;
            });
        });
    });
}


// SCROLL TO RESULTS AFTER FORM SUBMISSION
document.addEventListener("DOMContentLoaded", function () {

    animateSurvivalGauge();
    animateComparisonBars();

    const results = document.getElementById("results");
    const comparisonResults =
        document.getElementById("comparisonResults");

    if (results) {
        setTimeout(function () {
            results.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }, 350);

    } else if (comparisonResults) {
        setTimeout(function () {
            comparisonResults.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }, 350);
    }

    console.log("Titanic Survival Intelligence loaded successfully.");
});

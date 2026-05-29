document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('user-input-form');
    const resultContainer = document.getElementById('result-container');

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(data => {
            displayResults(data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });

    function displayResults(data) {
        resultContainer.innerHTML = `
            <h3>Your Digital Wellbeing Score: ${data.wellbeing_score}</h3>
            <h3>Smartphone Addiction Risk: ${data.addiction_risk}</h3>
            <h3>Productivity Level: ${data.productivity_level}</h3>
        `;
    }
});
function validateFileAndSubmit() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    const maxSize = 5 * 1024 * 1024;

    if (!file) {
        alert("Please select a file.");
        return false;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
        alert("Only CSV files are allowed.");
        return false;
    }

    if (file.size > maxSize) {
        alert("File size must be 5MB or less.");
        return false;
    }

    const btn = document.getElementById("submitBtn");
    const loadingText = document.getElementById("loadingText");
    btn.disabled = true;
    btn.innerText = "Processing...";
    loadingText.style.display = "inline";

    return true;
}

function createChurnChart(atRisk, safe, total) {
    const ctx = document.getElementById('churnChart').getContext('2d');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['At Risk', 'Not at Risk'],
            datasets: [{
                data: [atRisk, safe],
                backgroundColor: ['#dc3545', '#28a745'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const val = context.parsed;
                            const pct = ((val / total) * 100).toFixed(1);
                            return `${context.label}: ${val} (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}

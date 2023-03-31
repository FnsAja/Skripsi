let ctx = document.getElementById('barChart');
let label = [], accuracy = []

for(let i in all_fold){
    label.push(all_fold[i].fold)
    accuracy.push(all_fold[i].accuracy)
}

barChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: label,
        datasets: [{
            label: 'Accuracy',
            data: accuracy,
            borderWidth: 1
        }]
    },
    options: {
        plugin: {
            title: {
                display: true,
                text: 'Fold Score',
                padding: {
                    top: 10,
                    bottom: 30
                }
            },
            tooltip: {
                titleMarginBottom: 10,
                titleFontColor: '#6e707e',
                titleFontSize: 14,
                backgroundColor: "rgb(255,255,255)",
                bodyFontColor: "#858796",
                borderColor: '#dddfeb',
                borderWidth: 1,
                xPadding: 15,
                yPadding: 15,
                displayColors: false,
                caretPadding: 10,
                callbacks: {
                  label: function(tooltipItem, data) {
                    let label = data.labels[tooltipItem.index];
                    let val = data.datasets[tooltipItem.datasetIndex].data[tooltipItem.index];
                    let total = data.datasets[0].data.reduce((arr, curr) => arr + curr);
                    return label + ' : ' + val + ' (' + (val/total*100).toFixed(2) + '%)';
                  }
                }
            },
            scales: {
                x: {
                  gridLines: {
                    display: false,
                    drawBorder: false
                  },
                  maxBarThickness: 25,
                },
                y: {
                  ticks: {
                    beginAtZero: true,
                    padding: 10,
                  },
                  gridLines: {
                    color: "rgb(234, 236, 244)",
                    zeroLineColor: "rgb(234, 236, 244)",
                    drawBorder: true,
                    borderDash: [2],
                    zeroLineBorderDash: [2]
                  }
                },
            },
            legend: {
              display: false
            },
        },
    },
});
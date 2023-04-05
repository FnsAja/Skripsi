let count = [data['countNegative'], data['countNetral'], data['countPositive']] ?? []

ctx = document.getElementById("pieChart1");
pieChart1 = new Chart(ctx, {
  type: "pie",
  data: {
    labels: ["Negative", "Netral", "Positive"],
    datasets: [
      {
        data: [
          count[0],
          count[1],
          count[2],
        ],
        borderWidth: 1,
        borderColor: "#D3D3D3",
        backgroundColor: ["#9BD0F5", "#FFB1C1", "#77DD77"],
      },
    ],
  },
  options: {
    maintainAspectRatio: false,
    responsive: true,
    plugins: {
      legend: {
        labels: {
          usePointStyle: true,
        },
        position: "bottom",
      },
      title: {
        display: true,
        text: "Sentiment Test",
      },
      subtitle: {
        display: true,
        text: "Chart show the score of every sentiment",
        color: "blue",
        font: {
          size: 12,
          family: "tahoma",
          weight: "normal",
          style: "italic",
        },
        padding: {
          bottom: 10,
        },
      },
      tooltip: {
        titleColor: "#000",
        titleMarginBottom: 10,
        titleAlign: "center",
        backgroundColor: "#fff",
        bodyColor: "#808080",
        cornerRadius: 10,
        displayColors: false,
        callbacks: {
          title: (tooltipItems) => {
            return tooltipItems[0].dataset.label;
          },
          label: (tooltipItems) => {
            let total = tooltipItems.dataset.data.reduce(
              (arr, curr) => arr + curr
            );
            return `Jumlah - ${
              tooltipItems.dataset.data[tooltipItems.dataIndex]
            } (${(
              (tooltipItems.dataset.data[tooltipItems.dataIndex] / total) *
              100
            ).toFixed(2)}%)`;
          },
        },
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        display: false,
      },
    },
  },
});
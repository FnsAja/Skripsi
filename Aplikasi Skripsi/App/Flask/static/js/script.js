let label = [],
  accuracy = [],
  precision = [],
  recall = [],
  f1_score = [];

for (let i in all_fold) {
  label.push(all_fold[i].fold);
  accuracy.push(all_fold[i].accuracy);
  precision.push(all_fold[i].precision);
  recall.push(all_fold[i].recall);
  f1_score.push(all_fold[i].f1);
}

let ctx = document.getElementById("barChart");
barChart = new Chart(ctx, {
  type: "bar",
  data: {
    labels: label,
    datasets: [
      {
        label: "Accuracy",
        data: accuracy,
        borderWidth: 1,
        borderColor: "#D3D3D3",
        backgroundColor: "#9BD0F5",
      },
      {
        label: "Precision",
        data: precision,
        borderWidth: 1,
        borderColor: "#D3D3D3",
        backgroundColor: "#FFB1C1",
      },
      {
        label: "Recall",
        data: recall,
        borderWidth: 1,
        borderColor: "#D3D3D3",
        backgroundColor: "#77DD77",
      },
      {
        label: "F1 Score",
        data: f1_score,
        borderWidth: 1,
        borderColor: "#D3D3D3",
        backgroundColor: "#B19CD9",
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
        text: "10 Fold Cross Validation Score",
      },
      subtitle: {
        display: true,
        text: [
          "Chart show the score of every fold from 10 Fold Cross Validation",
          "Especially Precision, Recall, and F1-Score we use Macro Avg Cause of imbalance size of data per classes",
        ],
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
            return `Fold - ${tooltipItems.label} : ${(
              tooltipItems.dataset.data[tooltipItems.dataIndex] * 100
            ).toFixed(2)}%`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
        },
      },
      y: {
        grid: {
          display: true,
        },
      },
    },
  },
});

ctx = document.getElementById("pieChart");
barChart = new Chart(ctx, {
  type: "pie",
  data: {
    labels: ["Negative", "Netral", "Positive"],
    datasets: [
      {
        data: [1, 2, 3],
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
        text: "10 Fold Cross Validation Score",
      },
      subtitle: {
        display: true,
        text: [
          "Chart show the score of every fold from 10 Fold Cross Validation",
          "Especially Precision, Recall, and F1-Score we use Macro Avg Cause of imbalance size of data per classes",
        ],
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
            return `Fold - ${tooltipItems.label} : ${(
              tooltipItems.dataset.data[tooltipItems.dataIndex] * 100
            ).toFixed(2)}%`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          display: false,
        },
      },
      y: {
        grid: {
          display: false,
        },
      },
    },
  },
});

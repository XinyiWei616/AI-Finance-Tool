const API_BASE = "http://127.0.0.1:8000";
const tableBody = document.getElementById("table-body");
let myChart = null;

// --- A. 数据加载与渲染 ---

async function loadAllData() {
    try {
        const res = await fetch(`${API_BASE}/expenses`);
        const data = await res.json();
        
        console.log("DEBUG: 后端返回的第一条原始数据:", data[0]);

        const normalizedData = data.map(item => {
            // 使用更严谨的判断，确保只要有这个 Key 就用它的值
            const finalAmount = item.hasOwnProperty('amount') ? item.amount : (item.Amount || 0);
            const finalCategory = item.manual_category || item.category || item.Manual_Category || "OTHERS";
            const finalDesc = item.narrative || item.description || item.Narrative || "N/A";

            return {
                category: finalCategory,
                amount: parseFloat(finalAmount),
                date: item.date || "N/A",
                description: finalDesc,
                source: item.source || "AI_Import"
            };
        });

        renderTable(normalizedData);
        renderChart(normalizedData);
    } catch (err) {
        console.error("Failed to load data:", err);
    }
}
function renderTable(expenses) {
    tableBody.innerHTML = "";
    expenses.forEach(e => {
        const isAI = e.source === "AI_Import";
        const row = document.createElement("tr");
        if (isAI) row.classList.add("ai-row");

        row.innerHTML = `
            <td>${e.date}</td>
            <td>${e.description || 'N/A'}</td>
            <td>$${parseFloat(e.amount).toFixed(2)}</td>
            <td><strong>${e.category}</strong></td>
            <td>
                <span class="tag ${isAI ? 'tag-ai' : 'tag-manual'}">
                    ${isAI ? '🤖 AI' : '👤 Manual'}
                </span>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

function renderChart(expenses) {
    const ctx = document.getElementById('expenseChart').getContext('2d');
    
    // 按类别聚合数据
    const totals = expenses.reduce((acc, curr) => {
        const cat = curr.category || "Others";
        acc[cat] = (acc[cat] || 0) + parseFloat(curr.amount);
        return acc;
    }, {});

    if (myChart) myChart.destroy();
    myChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(totals),
            datasets: [{
                data: Object.values(totals),
                backgroundColor: ['#003566', '#000814', '#ffc300', '#ffd60a', '#001d3d'],
                borderWidth: 2
            }]
        },
        options: { plugins: { legend: { position: 'bottom' } } }
    });
}

// --- B. 交互逻辑 ---

// 1. 手动提交
document.getElementById("expense-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
        amount: parseFloat(document.getElementById("amount").value),
        category: document.getElementById("category").value,
        date: document.getElementById("date").value,
        description: document.getElementById("description").value
    };

    await fetch(`${API_BASE}/expenses`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    e.target.reset();
    loadAllData();
});

// 2. CSV 上传 (核心功能)
document.getElementById("upload-btn").addEventListener("click", async () => {
    const fileFile = document.getElementById("csv-file").files[0];
    if (!fileFile) return alert("Select a CSV file first.");

    const formData = new FormData();
    formData.append("file", fileFile);

    const res = await fetch(`${API_BASE}/upload-csv`, {
        method: "POST",
        body: formData
    });
    const result = await res.json();
    alert(`Successfully processed ${result.count} transactions!`);
    loadAllData();
});

// 3. 语义搜索
document.getElementById("search-btn").addEventListener("click", async () => {
    const query = document.getElementById("search-query").value;
    const res = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: query })
    });
    const result = await res.json();
    
    // 高亮展示搜索结果
    renderTable(result.data.map(item => ({
        ...item.metadata,
        source: "AI_Import",
        description: `🔍 [Match: ${(1-item.distance).toFixed(2)}] ${item.document}`
    })));
});

// 初始化加载
loadAllData();
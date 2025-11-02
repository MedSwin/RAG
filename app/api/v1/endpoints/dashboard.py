from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from app.services.hf_dataset_service import HuggingFaceDatasetService
from app.services.ingestion_pipeline_service import IngestionPipelineService
from app.services.model_download_service import ModelDownloadService

logger = logging.getLogger(__name__)
router = APIRouter()

class DatasetInfo(BaseModel):
    """Model for dataset information."""
    name: str
    description: str
    url: str
    repo_id: str
    expected_rows: int
    actual_rows: int
    size_gb: float
    status: str
    last_processed: Optional[str] = None
    processing_progress: float = 0

class IngestionRequest(BaseModel):
    """Request model for dataset ingestion."""
    dataset_name: str

class IngestionResponse(BaseModel):
    """Response model for dataset ingestion."""
    success: bool
    dataset_name: str
    message: str
    error: Optional[str] = None

class DashboardStats(BaseModel):
    """Model for dashboard statistics."""
    total_datasets: int
    total_rows: int
    total_size_gb: float
    status_counts: Dict[str, int]
    datasets: List[DatasetInfo]

def get_hf_service():
    """Dependency to get Hugging Face dataset service."""
    return HuggingFaceDatasetService()

def get_ingestion_service():
    """Dependency to get ingestion pipeline service."""
    return IngestionPipelineService()

def get_model_service():
    """Dependency to get model download service."""
    return ModelDownloadService()

@router.get("/", response_class=HTMLResponse)
async def dashboard_ui():
    """Serve the dashboard UI."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical RAG System Dashboard</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #2c3e50;
                overflow-x: hidden;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
                animation: fadeInDown 0.8s ease-out;
            }
            
            .header h1 {
                color: white;
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 10px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            
            .header p {
                color: rgba(255,255,255,0.9);
                font-size: 1.2rem;
                font-weight: 300;
            }
            
            .main-content {
                background: rgba(255,255,255,0.95);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
                animation: fadeInUp 0.8s ease-out 0.2s both;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 25px;
                margin-bottom: 50px;
            }
            
            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
                animation: slideInUp 0.6s ease-out;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 30px rgba(0,0,0,0.2);
            }
            
            .stat-card::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
                transform: rotate(45deg);
                transition: all 0.5s ease;
                opacity: 0;
            }
            
            .stat-card:hover::before {
                animation: shimmer 1.5s ease-in-out;
            }
            
            .stat-icon {
                font-size: 2.5rem;
                margin-bottom: 15px;
                opacity: 0.9;
            }
            
            .stat-number {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 8px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            
            .stat-label {
                font-size: 1rem;
                opacity: 0.9;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .datasets-section {
                margin-top: 40px;
            }
            
            .section-title {
                font-size: 2rem;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 30px;
                display: flex;
                align-items: center;
                gap: 15px;
            }
            
            .dataset-card {
                background: white;
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 25px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                border: 1px solid rgba(0,0,0,0.05);
                transition: all 0.3s ease;
                animation: slideInLeft 0.6s ease-out;
                position: relative;
                overflow: hidden;
            }
            
            .dataset-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 15px 30px rgba(0,0,0,0.15);
            }
            
            .dataset-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 4px;
                height: 100%;
                background: linear-gradient(135deg, #667eea, #764ba2);
                transition: width 0.3s ease;
            }
            
            .dataset-card:hover::before {
                width: 8px;
            }
            
            .dataset-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            
            .dataset-name {
                font-size: 1.4rem;
                font-weight: 600;
                color: #2c3e50;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .status-badge {
                padding: 8px 20px;
                border-radius: 25px;
                font-size: 0.85rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                position: relative;
                overflow: hidden;
            }
            
            .status-not-processed {
                background: linear-gradient(135deg, #e74c3c, #c0392b);
                color: white;
                box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
            }
            
            .status-processing {
                background: linear-gradient(135deg, #f39c12, #e67e22);
                color: white;
                box-shadow: 0 4px 15px rgba(243, 156, 18, 0.3);
                animation: pulse 2s infinite;
            }
            
            .status-processed {
                background: linear-gradient(135deg, #27ae60, #229954);
                color: white;
                box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
            }
            
            .status-error {
                background: linear-gradient(135deg, #8e44ad, #7d3c98);
                color: white;
                box-shadow: 0 4px 15px rgba(142, 68, 173, 0.3);
            }
            
            .dataset-description {
                color: #7f8c8d;
                margin-bottom: 25px;
                line-height: 1.6;
                font-size: 1rem;
            }
            
            .dataset-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 20px;
                margin-bottom: 25px;
            }
            
            .stat-item {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                border-radius: 10px;
                transition: all 0.3s ease;
                border: 1px solid rgba(0,0,0,0.05);
            }
            
            .stat-item:hover {
                background: linear-gradient(135deg, #e9ecef, #dee2e6);
                transform: translateY(-2px);
            }
            
            .stat-value {
                font-weight: 700;
                color: #2c3e50;
                font-size: 1.3rem;
                margin-bottom: 5px;
            }
            
            .stat-desc {
                font-size: 0.85rem;
                color: #7f8c8d;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .progress-container {
                margin-bottom: 25px;
                background: rgba(0,0,0,0.05);
                border-radius: 10px;
                overflow: hidden;
                height: 8px;
            }
            
            .progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #3498db, #2ecc71);
                border-radius: 10px;
                transition: width 0.8s ease;
                position: relative;
                overflow: hidden;
            }
            
            .progress-bar::after {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
                animation: progressShimmer 2s infinite;
            }
            
            .action-buttons {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }
            
            .btn {
                padding: 12px 25px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                transition: all 0.3s ease;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                position: relative;
                overflow: hidden;
            }
            
            .btn::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                background: rgba(255,255,255,0.2);
                border-radius: 50%;
                transform: translate(-50%, -50%);
                transition: all 0.3s ease;
            }
            
            .btn:hover::before {
                width: 300px;
                height: 300px;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
                box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
            }
            
            .btn-success {
                background: linear-gradient(135deg, #27ae60, #229954);
                color: white;
                box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
            }
            
            .btn-success:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(39, 174, 96, 0.4);
            }
            
            .btn-danger {
                background: linear-gradient(135deg, #e74c3c, #c0392b);
                color: white;
                box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
            }
            
            .btn-danger:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(231, 76, 60, 0.4);
            }
            
            .btn:disabled {
                background: #bdc3c7;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 40px 0;
                animation: fadeIn 0.5s ease;
            }
            
            .spinner {
                width: 50px;
                height: 50px;
                border: 4px solid rgba(255,255,255,0.3);
                border-top: 4px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            
            .loading-text {
                color: #7f8c8d;
                font-size: 1.1rem;
                font-weight: 500;
            }
            
            .message {
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                display: none;
                animation: slideInDown 0.5s ease;
                font-weight: 500;
            }
            
            .error-message {
                background: linear-gradient(135deg, #e74c3c, #c0392b);
                color: white;
                box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
            }
            
            .success-message {
                background: linear-gradient(135deg, #27ae60, #229954);
                color: white;
                box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
            }
            
            .refresh-indicator {
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(255,255,255,0.9);
                padding: 10px 15px;
                border-radius: 25px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                font-size: 0.85rem;
                color: #7f8c8d;
                display: none;
                animation: fadeIn 0.3s ease;
            }
            
            /* Animations */
            @keyframes fadeInDown {
                from {
                    opacity: 0;
                    transform: translateY(-30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(50px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes slideInLeft {
                from {
                    opacity: 0;
                    transform: translateX(-50px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            @keyframes slideInDown {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            
            @keyframes shimmer {
                0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
                100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
            }
            
            @keyframes progressShimmer {
                0% { left: -100%; }
                100% { left: 100%; }
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                .main-content {
                    padding: 20px;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .stats-grid {
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                }
                
                .dataset-stats {
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 15px;
                }
                
                .action-buttons {
                    flex-direction: column;
                }
                
                .btn {
                    justify-content: center;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-hospital"></i> Medical RAG System</h1>
                <p>Advanced Document Processing & Retrieval Dashboard</p>
            </div>
            
            <div class="main-content">
                <div class="stats-grid" id="stats-grid">
                    <div class="stat-card" style="animation-delay: 0.1s;">
                        <div class="stat-icon"><i class="fas fa-database"></i></div>
                        <div class="stat-number" id="total-datasets">-</div>
                        <div class="stat-label">Total Datasets</div>
                    </div>
                    <div class="stat-card" style="animation-delay: 0.2s;">
                        <div class="stat-icon"><i class="fas fa-file-alt"></i></div>
                        <div class="stat-number" id="total-rows">-</div>
                        <div class="stat-label">Total Rows</div>
                    </div>
                    <div class="stat-card" style="animation-delay: 0.3s;">
                        <div class="stat-icon"><i class="fas fa-hdd"></i></div>
                        <div class="stat-number" id="total-size">-</div>
                        <div class="stat-label">Size (GB)</div>
                    </div>
                    <div class="stat-card" style="animation-delay: 0.4s;">
                        <div class="stat-icon"><i class="fas fa-check-circle"></i></div>
                        <div class="stat-number" id="processed-count">-</div>
                        <div class="stat-label">Processed</div>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <div class="loading-text">Loading dashboard data...</div>
                </div>
                
                <div class="message error-message" id="error-message"></div>
                <div class="message success-message" id="success-message"></div>
                
                <div class="datasets-section">
                    <div class="section-title">
                        <i class="fas fa-chart-bar"></i>
                        Dataset Management
                    </div>
                    <div id="datasets-container">
                        <!-- Datasets will be loaded here -->
                    </div>
                </div>
            </div>
        </div>
        
        <div class="refresh-indicator" id="refresh-indicator">
            <i class="fas fa-sync-alt"></i> Auto-refreshing...
        </div>

        <script>
            let datasets = [];
            let refreshInterval;
            
            async function loadDashboard() {
                showLoading(true);
                try {
                    const response = await fetch('/api/v1/dashboard/stats');
                    const data = await response.json();
                    
                    updateStats(data);
                    datasets = data.datasets;
                    renderDatasets();
                    
                } catch (error) {
                    showError('Failed to load dashboard data: ' + error.message);
                } finally {
                    showLoading(false);
                }
            }
            
            function updateStats(data) {
                animateNumber('total-datasets', data.total_datasets);
                animateNumber('total-rows', data.total_rows);
                animateNumber('total-size', data.total_size_gb);
                animateNumber('processed-count', data.status_counts.processed || 0);
            }
            
            function animateNumber(elementId, targetValue) {
                const element = document.getElementById(elementId);
                const currentValue = parseInt(element.textContent) || 0;
                const increment = (targetValue - currentValue) / 20;
                let current = currentValue;
                
                const timer = setInterval(() => {
                    current += increment;
                    if ((increment > 0 && current >= targetValue) || (increment < 0 && current <= targetValue)) {
                        current = targetValue;
                        clearInterval(timer);
                    }
                    
                    if (elementId === 'total-rows' || elementId === 'processed-count') {
                        element.textContent = Math.floor(current).toLocaleString();
                    } else if (elementId === 'total-size') {
                        element.textContent = current.toFixed(1);
                    } else {
                        element.textContent = Math.floor(current);
                    }
                }, 50);
            }
            
            function renderDatasets() {
                const container = document.getElementById('datasets-container');
                container.innerHTML = '';
                
                datasets.forEach((dataset, index) => {
                    setTimeout(() => {
                        const datasetCard = createDatasetCard(dataset);
                        container.appendChild(datasetCard);
                    }, index * 100);
                });
            }
            
            function createDatasetCard(dataset) {
                const card = document.createElement('div');
                card.className = 'dataset-card';
                
                const statusClass = `status-${dataset.status.replace('_', '-')}`;
                const progress = dataset.processing_progress || 0;
                
                card.innerHTML = `
                    <div class="dataset-header">
                        <div class="dataset-name">
                            <i class="fas fa-database"></i>
                            ${dataset.name}
                        </div>
                        <div class="status-badge ${statusClass}">${dataset.status}</div>
                    </div>
                    <div class="dataset-description">${dataset.description}</div>
                    <div class="dataset-stats">
                        <div class="stat-item">
                            <div class="stat-value">${dataset.actual_rows.toLocaleString()}</div>
                            <div class="stat-desc">Rows</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${dataset.size_gb} GB</div>
                            <div class="stat-desc">Size</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${dataset.expected_rows.toLocaleString()}</div>
                            <div class="stat-desc">Expected</div>
                        </div>
                    </div>
                    ${dataset.status === 'processing' ? `
                        <div class="progress-container">
                            <div class="progress-bar" style="width: ${progress}%"></div>
                        </div>
                    ` : ''}
                    <div class="action-buttons">
                        ${dataset.status === 'not_processed' ? `
                            <button class="btn btn-primary" onclick="ingestDataset('${dataset.name}')">
                                <i class="fas fa-rocket"></i>
                                Start Ingestion
                            </button>
                        ` : ''}
                        ${dataset.status === 'processed' ? `
                            <button class="btn btn-success" onclick="viewDataset('${dataset.name}')">
                                <i class="fas fa-eye"></i>
                                View Data
                            </button>
                        ` : ''}
                        <a href="${dataset.url}" target="_blank" class="btn btn-primary">
                            <i class="fas fa-external-link-alt"></i>
                            View on HF
                        </a>
                    </div>
                `;
                
                return card;
            }
            
            async function ingestDataset(datasetName) {
                showLoading(true);
                try {
                    const response = await fetch(`/api/v1/dashboard/ingest/${datasetName}`, {
                        method: 'POST'
                    });
                    const result = await response.json();
                    
                    if (result.success) {
                        showSuccess(`Started ingestion for ${datasetName}`);
                        setTimeout(() => {
                            loadDashboard();
                        }, 2000);
                    } else {
                        showError(`Failed to start ingestion: ${result.error}`);
                    }
                } catch (error) {
                    showError('Failed to start ingestion: ' + error.message);
                } finally {
                    showLoading(false);
                }
            }
            
            function viewDataset(datasetName) {
                window.open(`/api/v1/dashboard/dataset/${datasetName}`, '_blank');
            }
            
            function showLoading(show) {
                document.getElementById('loading').style.display = show ? 'block' : 'none';
            }
            
            function showError(message) {
                const errorDiv = document.getElementById('error-message');
                errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
                errorDiv.style.display = 'block';
                setTimeout(() => {
                    errorDiv.style.display = 'none';
                }, 5000);
            }
            
            function showSuccess(message) {
                const successDiv = document.getElementById('success-message');
                successDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
                successDiv.style.display = 'block';
                setTimeout(() => {
                    successDiv.style.display = 'none';
                }, 5000);
            }
            
            function showRefreshIndicator() {
                const indicator = document.getElementById('refresh-indicator');
                indicator.style.display = 'block';
                setTimeout(() => {
                    indicator.style.display = 'none';
                }, 2000);
            }
            
            // Load dashboard on page load
            loadDashboard();
            
            // Auto-refresh every 5 minutes (300000ms) to avoid timeout issues
            refreshInterval = setInterval(() => {
                showRefreshIndicator();
                loadDashboard();
            }, 300000);
            
            // Cleanup on page unload
            window.addEventListener('beforeunload', () => {
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(hf_service = Depends(get_hf_service)):
    """Get dashboard statistics."""
    import asyncio
    try:
        # Use cache if available, otherwise load with short timeout
        # Cache is populated during server startup preloading
        stats = await asyncio.wait_for(
            hf_service.get_total_statistics(use_cache=True),
            timeout=10.0  # 10 second timeout (should be instant if cached)
        )
        
        # Convert to response format
        datasets = []
        for dataset_info in stats["datasets"]:
            datasets.append(DatasetInfo(**dataset_info))
        
        return DashboardStats(
            total_datasets=stats["total_datasets"],
            total_rows=stats["total_rows"],
            total_size_gb=stats["total_size_gb"],
            status_counts=stats["status_counts"],
            datasets=datasets
        )
        
    except asyncio.TimeoutError:
        logger.error("Timeout getting dashboard stats")
        # Return empty stats instead of error
        return DashboardStats(
            total_datasets=0,
            total_rows=0,
            total_size_gb=0.0,
            status_counts={},
            datasets=[]
        )
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        # Return empty stats instead of error
        return DashboardStats(
            total_datasets=0,
            total_rows=0,
            total_size_gb=0.0,
            status_counts={},
            datasets=[]
        )

@router.post("/ingest/{dataset_name}", response_model=IngestionResponse)
async def start_dataset_ingestion(
    dataset_name: str,
    background_tasks: BackgroundTasks,
    ingestion_service = Depends(get_ingestion_service)
):
    """Start ingestion pipeline for a dataset."""
    try:
        # Start ingestion in background
        background_tasks.add_task(
            ingestion_service.run_full_ingestion_pipeline,
            dataset_name
        )
        
        return IngestionResponse(
            success=True,
            dataset_name=dataset_name,
            message=f"Ingestion started for {dataset_name}"
        )
        
    except Exception as e:
        logger.error(f"Error starting ingestion for {dataset_name}: {e}")
        return IngestionResponse(
            success=False,
            dataset_name=dataset_name,
            error=str(e)
        )

@router.get("/dataset/{dataset_name}")
async def get_dataset_details(
    dataset_name: str,
    hf_service = Depends(get_hf_service)
):
    """Get detailed information about a specific dataset."""
    try:
        dataset_info = await hf_service.get_dataset_info(dataset_name)
        return dataset_info
        
    except Exception as e:
        logger.error(f"Error getting dataset details for {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset details: {str(e)}")

@router.get("/status")
async def get_ingestion_statuses(ingestion_service = Depends(get_ingestion_service)):
    """Get ingestion status for all datasets."""
    try:
        statuses = await ingestion_service.get_all_ingestion_statuses()
        return {"statuses": statuses}
        
    except Exception as e:
        logger.error(f"Error getting ingestion statuses: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ingestion statuses: {str(e)}")

@router.get("/models")
async def get_models_info(model_service = Depends(get_model_service)):
    """Get information about downloaded models."""
    try:
        models_info = await model_service.get_all_models_info()
        return {"models": models_info}
        
    except Exception as e:
        logger.error(f"Error getting models info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models info: {str(e)}")

@router.post("/download-models")
async def download_models(model_service = Depends(get_model_service)):
    """Download all required models."""
    try:
        results = await model_service.download_all_models()
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download models: {str(e)}")

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const askForm = document.getElementById('ask-form');
    const questionInput = document.getElementById('question-input');
    const topKSelect = document.getElementById('top_k-select');
    const topKVal = document.getElementById('top_k-val');
    const toggleDiagnostics = document.getElementById('toggle-diagnostics');
    const submitBtn = document.getElementById('submit-btn');
    const connectionStatus = document.getElementById('connection-status');
    
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    const retryBtn = document.getElementById('retry-btn');
    
    const loadingContainer = document.getElementById('loading-container');
    const answerContainer = document.getElementById('answer-container');
    const answerText = document.getElementById('answer-text');
    const clearBtn = document.getElementById('clear-btn');
    
    const sourcesContainer = document.getElementById('sources-container');
    const sourcesList = document.getElementById('sources-list');
    
    const diagnosticsContainer = document.getElementById('diagnostics-container');
    const diagnosticsToggle = document.getElementById('diagnostics-toggle');
    const diagnosticsBody = document.getElementById('diagnostics-body');
    
    const diagRetrievalMode = document.getElementById('diag-retrieval-mode');
    const diagGenerationMode = document.getElementById('diag-generation-mode');
    const diagTopK = document.getElementById('diag-top-k');
    const diagResponseTime = document.getElementById('diag-response-time');
    
    const emptyState = document.getElementById('empty-state');
    
    // Update top_k value label
    topKSelect.addEventListener('input', (e) => {
        topKVal.textContent = e.target.value;
    });
    
    // Toggle Diagnostics Panel open/close
    diagnosticsToggle.addEventListener('click', () => {
        diagnosticsContainer.classList.toggle('collapsed');
    });

    // Determine the base API URL dynamically (supports file:// and FastAPI server)
    const getApiUrl = () => {
        if (window.location.protocol === 'file:' || window.location.hostname === '') {
            return 'http://127.0.0.1:8000/ask';
        }
        return '/ask';
    };

    // Check backend connection status
    const checkConnection = async () => {
        const statusText = connectionStatus.querySelector('.status-text');
        
        let healthUrl;
        if (window.location.protocol === 'file:' || window.location.hostname === '') {
            healthUrl = 'http://127.0.0.1:8000/health';
        } else {
            healthUrl = '/health';
        }

        try {
            const response = await fetch(healthUrl, { method: 'GET' });
            if (response.ok) {
                connectionStatus.className = 'status-indicator connected';
                statusText.textContent = 'Backend connected';
            } else {
                throw new Error('Not OK response');
            }
        } catch (err) {
            connectionStatus.className = 'status-indicator disconnected';
            statusText.textContent = 'Backend unavailable';
        }
    };

    // Clean UI state
    const resetUI = () => {
        answerContainer.classList.add('hidden');
        sourcesContainer.classList.add('hidden');
        diagnosticsContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
        emptyState.classList.remove('hidden');
    };

    clearBtn.addEventListener('click', () => {
        questionInput.value = '';
        resetUI();
    });

    // Form Submission
    const submitQuestion = async (queryText) => {
        if (!queryText.trim()) return;

        // UI States: loading active, others hidden
        emptyState.classList.add('hidden');
        errorContainer.classList.add('hidden');
        answerContainer.classList.add('hidden');
        sourcesContainer.classList.add('hidden');
        diagnosticsContainer.classList.add('hidden');
        
        loadingContainer.classList.remove('hidden');
        submitBtn.disabled = true;
        
        const top_k = parseInt(topKSelect.value, 10);
        
        try {
            const response = await fetch(getApiUrl(), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: queryText,
                    top_k: top_k
                })
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Mark connected
            connectionStatus.className = 'status-indicator connected';
            connectionStatus.querySelector('.status-text').textContent = 'Backend connected';
            
            // Hide loading
            loadingContainer.classList.add('hidden');
            
            // Render Answer
            renderAnswer(data.answer);
            answerContainer.classList.remove('hidden');
            
            // Render Sources
            if (data.sources && data.sources.length > 0) {
                renderSources(data.sources);
                sourcesContainer.classList.remove('hidden');
            } else {
                sourcesList.innerHTML = '<p class="text-muted">No source files were retrieved for this answer.</p>';
                sourcesContainer.classList.remove('hidden');
            }
            
            // Render Diagnostics if toggle is checked
            if (toggleDiagnostics.checked && data.metadata) {
                renderDiagnostics(data.metadata);
                diagnosticsContainer.classList.remove('hidden');
                diagnosticsContainer.classList.remove('collapsed'); // show open by default when diagnostic toggle is enabled
            }
            
        } catch (err) {
            console.error('Error fetching RAG response:', err);
            // Mark disconnected
            connectionStatus.className = 'status-indicator disconnected';
            connectionStatus.querySelector('.status-text').textContent = 'Backend unavailable';
            
            loadingContainer.classList.add('hidden');
            errorMessage.textContent = `Could not get a response from the local RAG backend. details: ${err.message}. Ensure python backend is running.`;
            errorContainer.classList.remove('hidden');
        } finally {
            submitBtn.disabled = false;
        }
    };

    askForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitQuestion(questionInput.value);
    });

    retryBtn.addEventListener('click', () => {
        checkConnection();
        submitQuestion(questionInput.value);
    });

    // Make citations clickable in the answer text and scroll to corresponding source card
    const renderAnswer = (rawText) => {
        // Find citations like [Source 1], [Source 2], etc.
        const citationRegex = /\[Source\s+(\d+)\]/gi;
        const htmlText = rawText.replace(citationRegex, (match, num) => {
            return `<span class="citation" data-source="${num}">[Source ${num}]</span>`;
        });
        
        answerText.innerHTML = htmlText;
        
        // Attach click listeners to citation tags
        answerText.querySelectorAll('.citation').forEach(citeSpan => {
            citeSpan.addEventListener('click', () => {
                const sourceNum = citeSpan.getAttribute('data-source');
                const sourceCard = document.getElementById(`source-card-${sourceNum}`);
                if (sourceCard) {
                    sourceCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    // Visual pulse effect
                    sourceCard.style.borderColor = '#a78bfa';
                    sourceCard.style.boxShadow = '0 0 15px rgba(139, 92, 246, 0.4)';
                    setTimeout(() => {
                        sourceCard.style.borderColor = '';
                        sourceCard.style.boxShadow = '';
                    }, 2000);
                }
            });
        });
    };

    // Render source cards
    const renderSources = (sources) => {
        sourcesList.innerHTML = '';
        sources.forEach((src, idx) => {
            const sourceIndex = idx + 1;
            const file = src.metadata.file_name || 'unknown_file';
            const chunkIdx = src.metadata.chunk_index !== undefined ? src.metadata.chunk_index : 'N/A';
            const scoreVal = src.score !== null && src.score !== undefined ? src.score.toFixed(4) : null;
            
            const card = document.createElement('div');
            card.className = 'source-item';
            card.id = `source-card-${sourceIndex}`;
            
            const scoreHtml = scoreVal ? `<span class="source-score">Relevance metric: ${scoreVal}</span>` : '';
            
            card.innerHTML = `
                <div class="source-item-header">
                    <div class="source-name-badge">
                        <span class="source-number">Source ${sourceIndex}</span>
                        <span class="source-file">${file} (Chunk ${chunkIdx})</span>
                    </div>
                    ${scoreHtml}
                </div>
                <div class="source-text">${escapeHtml(src.text)}</div>
            `;
            sourcesList.appendChild(card);
        });
    };

    // Populate diagnostics data
    const renderDiagnostics = (meta) => {
        diagRetrievalMode.textContent = meta.retrieval_mode || 'Hybrid (Default)';
        diagGenerationMode.textContent = meta.generation_mode || 'Local LLM';
        diagTopK.textContent = meta.top_k_used !== undefined ? meta.top_k_used : '3';
        diagResponseTime.textContent = meta.response_time_ms !== undefined ? `${meta.response_time_ms} ms` : 'N/A';
    };

    // Helper: Escapes HTML characters
    const escapeHtml = (text) => {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    };

    // Connect example question buttons
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.getAttribute('data-question');
            questionInput.value = question;
            submitQuestion(question);
        });
    });

    // Check connection on page load
    checkConnection();
});

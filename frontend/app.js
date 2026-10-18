/**
 * Chess Analysis Coach — Frontend Application
 */

(function () {
  'use strict';

  // Sample game with clear blunders/tactical mistakes
  const SAMPLE_PGN = `1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d3 8. Qb3 Qf6 9. e5 Qg6 10. Re1 Nge7 11. Ba3 b5 12. Qxb5 Rb8 13. Qa4 Bb6 14. Nbd2 Bb7 15. Ne4 O-O 16. Bxd3 Qh5 17. Ng3 Qh6 18. Bc1 Qe6 19. Bxh7+ Kxh7 20. Ng5+ Kg8 21. Nxe6 fxe6 22. Be3`;

  // State
  let board = null;
  let gameHistory = []; // { moveIndex, moveNumber, side, san, fen, beforeFen }
  let currentIndex = -1; // -1 = start position
  let explanationsMap = {}; // "2_white" -> MoveExplanation
  let pollTimer = null;

  // DOM Elements
  const pgnInput = document.getElementById('pgn-input');
  const depthInput = document.getElementById('depth-input');
  const btnAnalyze = document.getElementById('btn-analyze');
  const btnSample = document.getElementById('btn-sample');
  const analyzeSpinner = document.getElementById('analyze-spinner');
  const analyzeBtnText = document.getElementById('analyze-btn-text');
  const statusBanner = document.getElementById('status-banner');
  const statusIcon = document.getElementById('status-icon');
  const statusText = document.getElementById('status-text');

  const statsBar = document.getElementById('stats-bar');
  const statTotal = document.getElementById('stat-total');
  const statBlunders = document.getElementById('stat-blunders');
  const statMistakes = document.getElementById('stat-mistakes');
  const statInaccuracies = document.getElementById('stat-inaccuracies');

  const moveListContainer = document.getElementById('move-list');
  const moveIndicator = document.getElementById('move-indicator');

  const btnFirst = document.getElementById('btn-first');
  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');
  const btnLast = document.getElementById('btn-last');
  const btnFlip = document.getElementById('btn-flip');

  const moveClassBadge = document.getElementById('move-class-badge');
  const evalShiftBadge = document.getElementById('eval-shift-badge');
  const moveComparison = document.getElementById('move-comparison');
  const playedMoveText = document.getElementById('played-move-text');
  const bestMoveText = document.getElementById('best-move-text');
  const explanationContent = document.getElementById('explanation-content');
  const sourcesContainer = document.getElementById('sources-container');
  const sourcesTags = document.getElementById('sources-tags');

  // Initialize Chessboard
  function initBoard() {
    const config = {
      draggable: false,
      position: 'start',
      pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
    };
    board = ChessBoard('board', config);
    $(window).resize(board.resize);
  }

  // Parse PGN using chess.js
  function parsePgnMoves(pgnText) {
    const chess = new Chess();
    const success = chess.load_pgn(pgnText);
    if (!success) {
      throw new Error('Invalid PGN format. Please verify the game notation.');
    }

    const history = chess.history({ verbose: true });
    const moves = [];
    const replay = new Chess();

    for (let i = 0; i < history.length; i++) {
      const beforeFen = replay.fen();
      const h = history[i];
      replay.move(h);
      const afterFen = replay.fen();

      const moveNumber = Math.floor(i / 2) + 1;
      const side = (i % 2 === 0) ? 'white' : 'black';

      moves.push({
        moveIndex: i,
        moveNumber: moveNumber,
        side: side,
        san: h.san,
        beforeFen: beforeFen,
        fen: afterFen
      });
    }

    return moves;
  }

  // Render Move List
  function renderMoveList() {
    if (gameHistory.length === 0) {
      moveListContainer.innerHTML = '<div class="empty-hint">Moves will appear here after loading a game.</div>';
      return;
    }

    let html = '';
    const totalMoves = Math.ceil(gameHistory.length / 2);

    for (let m = 1; m <= totalMoves; m++) {
      const whiteIdx = (m - 1) * 2;
      const blackIdx = whiteIdx + 1;

      const whiteMove = gameHistory[whiteIdx];
      const blackMove = blackIdx < gameHistory.length ? gameHistory[blackIdx] : null;

      html += `<div class="move-row">`;
      html += `<span class="move-num">${m}.</span>`;

      // White button
      const wKey = `${whiteMove.moveNumber}_white`;
      const wExp = explanationsMap[wKey];
      const wClass = wExp ? `is-${wExp.classification}` : '';
      const wActive = (currentIndex === whiteIdx) ? 'active' : '';
      html += `<button class="move-btn ${wClass} ${wActive}" data-index="${whiteIdx}">${whiteMove.san}</button>`;

      // Black button
      if (blackMove) {
        const bKey = `${blackMove.moveNumber}_black`;
        const bExp = explanationsMap[bKey];
        const bClass = bExp ? `is-${bExp.classification}` : '';
        const bActive = (currentIndex === blackIdx) ? 'active' : '';
        html += `<button class="move-btn ${bClass} ${bActive}" data-index="${blackIdx}">${blackMove.san}</button>`;
      } else {
        html += `<span></span>`;
      }

      html += `</div>`;
    }

    moveListContainer.innerHTML = html;

    // Attach click listeners to move buttons
    const buttons = moveListContainer.querySelectorAll('.move-btn');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.getAttribute('data-index'), 10);
        goToMove(idx);
      });
    });
  }

  // Go to Move
  function goToMove(index) {
    if (gameHistory.length === 0) return;

    if (index < -1) index = -1;
    if (index >= gameHistory.length) index = gameHistory.length - 1;

    currentIndex = index;

    // Update board position
    if (index === -1) {
      board.start();
      moveIndicator.textContent = 'Start of game';
    } else {
      const item = gameHistory[index];
      board.position(item.fen);
      const sideCap = item.side.charAt(0).toUpperCase() + item.side.slice(1);
      moveIndicator.textContent = `Move ${item.moveNumber}. ${sideCap} (${item.san})`;
    }

    // Update active highlight in move list
    const buttons = moveListContainer.querySelectorAll('.move-btn');
    buttons.forEach(b => {
      const bIdx = parseInt(b.getAttribute('data-index'), 10);
      if (bIdx === currentIndex) {
        b.classList.add('active');
        b.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      } else {
        b.classList.remove('active');
      }
    });

    // Update Explanation Card
    updateExplanationView();
  }

  // Update Explanation View for current position
  function updateExplanationView() {
    if (currentIndex === -1) {
      moveClassBadge.style.display = 'none';
      evalShiftBadge.style.display = 'none';
      moveComparison.style.display = 'none';
      sourcesContainer.style.display = 'none';
      explanationContent.innerHTML = '<p class="placeholder-text">Initial position. Step forward or click a move to inspect the game.</p>';
      return;
    }

    const currentMove = gameHistory[currentIndex];
    const key = `${currentMove.moveNumber}_${currentMove.side}`;
    const exp = explanationsMap[key];

    if (exp) {
      // Flagged move
      moveClassBadge.style.display = 'inline-flex';
      moveClassBadge.textContent = exp.classification.toUpperCase();
      moveClassBadge.className = `badge badge-${exp.classification}`;

      evalShiftBadge.style.display = 'inline-block';
      evalShiftBadge.textContent = `Eval: ${exp.eval_before} → ${exp.eval_after}`;

      moveComparison.style.display = 'grid';
      playedMoveText.textContent = exp.played;
      bestMoveText.textContent = exp.best;

      explanationContent.textContent = exp.explanation;

      // Citations
      if (exp.sources && exp.sources.length > 0) {
        sourcesContainer.style.display = 'flex';
        sourcesTags.innerHTML = exp.sources
          .map(s => `<span class="source-tag">${s}</span>`)
          .join('');
      } else {
        sourcesContainer.style.display = 'none';
      }
    } else {
      // Good move
      moveClassBadge.style.display = 'inline-flex';
      moveClassBadge.textContent = 'GOOD MOVE';
      moveClassBadge.className = 'badge badge-good';

      evalShiftBadge.style.display = 'none';
      moveComparison.style.display = 'none';
      sourcesContainer.style.display = 'none';

      explanationContent.innerHTML = `<p>The move <strong>${currentMove.san}</strong> is sound. Stockfish did not detect any significant evaluation loss.</p>`;
    }
  }

  // Update UI Status Banner
  function setStatus(type, message) {
    statusBanner.className = `status-banner status-${type}`;
    statusText.textContent = message;
    if (type === 'running') {
      analyzeSpinner.style.display = 'inline-block';
      btnAnalyze.disabled = true;
      analyzeBtnText.textContent = 'Analyzing...';
    } else {
      analyzeSpinner.style.display = 'none';
      btnAnalyze.disabled = false;
      analyzeBtnText.textContent = 'Analyze Game';
    }
  }

  // Start Game Analysis
  async function startAnalysis() {
    const pgn = pgnInput.value.trim();
    if (!pgn) {
      setStatus('error', 'Please enter or paste a PGN game to analyze.');
      return;
    }

    // Validate and load moves into chess.js
    try {
      gameHistory = parsePgnMoves(pgn);
    } catch (err) {
      setStatus('error', err.message);
      return;
    }

    explanationsMap = {};
    renderMoveList();
    goToMove(-1);

    const depth = parseInt(depthInput.value, 10) || 14;
    setStatus('running', 'Submitting game to Stockfish engine...');

    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pgn: pgn, depth: depth })
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned error ${resp.status}`);
      }

      const created = await resp.json();
      const jobId = created.job_id;

      setStatus('running', 'Analyzing positions with Stockfish and generating AI coaching explanations...');
      pollJobStatus(jobId);
    } catch (err) {
      setStatus('error', `Failed to start analysis: ${err.message}`);
    }
  }

  // Poll Job Status
  function pollJobStatus(jobId) {
    if (pollTimer) clearInterval(pollTimer);

    pollTimer = setInterval(async () => {
      try {
        const resp = await fetch(`/api/analyze/${jobId}`);
        if (!resp.ok) {
          throw new Error(`Failed to check job status (${resp.status})`);
        }

        const data = await resp.json();

        if (data.status === 'completed') {
          clearInterval(pollTimer);
          pollTimer = null;
          handleAnalysisSuccess(data.result);
        } else if (data.status === 'failed') {
          clearInterval(pollTimer);
          pollTimer = null;
          setStatus('error', `Analysis failed: ${data.error || 'Unknown error'}`);
        } else {
          // pending or running
          setStatus('running', `Engine analyzing positions (job: ${jobId.slice(0, 8)})...`);
        }
      } catch (err) {
        clearInterval(pollTimer);
        pollTimer = null;
        setStatus('error', `Polling error: ${err.message}`);
      }
    }, 1200);
  }

  // Handle successful analysis
  function handleAnalysisSuccess(result) {
    setStatus('success', `Analysis complete! Flagged ${result.flagged_moves} mistake(s) across ${result.total_moves} moves.`);

    let blunders = 0;
    let mistakes = 0;
    let inaccuracies = 0;

    explanationsMap = {};
    for (const exp of result.explanations) {
      const key = `${exp.move_number}_${exp.side}`;
      explanationsMap[key] = exp;
      if (exp.classification === 'blunder') blunders++;
      else if (exp.classification === 'mistake') mistakes++;
      else if (exp.classification === 'inaccuracy') inaccuracies++;
    }

    // Update stats bar
    statTotal.textContent = result.total_moves;
    statBlunders.textContent = blunders;
    statMistakes.textContent = mistakes;
    statInaccuracies.textContent = inaccuracies;
    statsBar.style.display = 'grid';

    // Re-render move list with colored badges
    renderMoveList();

    // Find and jump to first flagged mistake
    let firstMistakeIdx = -1;
    for (let i = 0; i < gameHistory.length; i++) {
      const item = gameHistory[i];
      const key = `${item.moveNumber}_${item.side}`;
      if (explanationsMap[key]) {
        firstMistakeIdx = i;
        break;
      }
    }

    goToMove(firstMistakeIdx !== -1 ? firstMistakeIdx : 0);
  }

  // Keyboard navigation
  window.addEventListener('keydown', (e) => {
    // Ignore keys when typing in textarea or inputs
    if (['TEXTAREA', 'INPUT'].includes(document.activeElement.tagName)) return;

    if (e.key === 'ArrowLeft') {
      goToMove(currentIndex - 1);
    } else if (e.key === 'ArrowRight') {
      goToMove(currentIndex + 1);
    } else if (e.key === 'Home') {
      goToMove(-1);
    } else if (e.key === 'End') {
      goToMove(gameHistory.length - 1);
    }
  });

  // Event Listeners
  btnAnalyze.addEventListener('click', startAnalysis);

  btnSample.addEventListener('click', () => {
    pgnInput.value = SAMPLE_PGN;
    setStatus('idle', 'Sample game loaded. Click "Analyze Game" to review.');
  });

  btnFirst.addEventListener('click', () => goToMove(-1));
  btnPrev.addEventListener('click', () => goToMove(currentIndex - 1));
  btnNext.addEventListener('click', () => goToMove(currentIndex + 1));
  btnLast.addEventListener('click', () => goToMove(gameHistory.length - 1));
  btnFlip.addEventListener('click', () => board.flip());


  // =========================================================================
  // TAB NAVIGATION & WEAKNESS PROFILE
  // =========================================================================

  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        if (targetId === 'tab-analyze') {
          targetPane.classList.add('active');
          if (board) board.resize();
        } else {
          targetPane.style.display = 'flex';
        }
      }
      tabPanes.forEach(p => {
        if (p.id !== targetId) {
          p.classList.remove('active');
          p.style.display = 'none';
        }
      });
    });
  });

  // Weakness elements
  const weaknessPlayerInput = document.getElementById('weakness-player-input');
  const btnFetchWeaknesses = document.getElementById('btn-fetch-weaknesses');
  const btnImportLichess = document.getElementById('btn-import-lichess');
  const weaknessSpinner = document.getElementById('weakness-spinner');
  const lichessSpinner = document.getElementById('lichess-spinner');
  const weaknessStatus = document.getElementById('weakness-status');
  const weaknessResults = document.getElementById('weakness-results');
  const overallRateBadge = document.getElementById('overall-rate-badge');
  const weaknessTableBody = document.getElementById('weakness-table-body');
  const weaknessCoachingText = document.getElementById('weakness-coaching-text');
  const weaknessSourcesContainer = document.getElementById('weakness-sources-container');
  const weaknessSourcesTags = document.getElementById('weakness-sources-tags');

  async function loadWeaknessProfile() {
    const player = weaknessPlayerInput.value.trim();
    if (!player) {
      weaknessStatus.className = 'status-banner status-error';
      weaknessStatus.textContent = 'Please enter a player username.';
      return;
    }

    weaknessSpinner.style.display = 'inline-block';
    btnFetchWeaknesses.disabled = true;
    weaknessStatus.className = 'status-banner status-running';
    weaknessStatus.textContent = `Analyzing statistical weakness profile for ${player}...`;

    try {
      const resp = await fetch(`/api/weaknesses?player=${encodeURIComponent(player)}`);
      if (!resp.ok) {
        throw new Error(`Server returned error ${resp.status}`);
      }
      const data = await resp.json();

      if (data.status === 'not_enough_games') {
        weaknessStatus.className = 'status-banner status-idle';
        weaknessStatus.textContent = data.message || 'Need more games analyzed.';
        weaknessResults.style.display = 'none';
        return;
      }

      weaknessStatus.className = 'status-banner status-success';
      weaknessStatus.textContent = `Analyzed ${data.total_moves} moves across ${data.total_games} games for ${data.player}.`;
      overallRateBadge.textContent = `Overall Error Rate: ${(data.overall_error_rate * 100).toFixed(1)}%`;

      // Render Table
      let tableHtml = '';
      for (const w of data.top_weaknesses) {
        tableHtml += `<tr>
          <td><strong>${w.category}</strong></td>
          <td><span class="badge badge-rag">${w.dimension}</span></td>
          <td>${w.total_moves}</td>
          <td style="color: var(--color-blunder);">${w.mistakes_and_blunders}</td>
          <td>${(w.error_rate * 100).toFixed(1)}%</td>
          <td style="color: var(--color-mistake); font-weight: 700;">+${(w.delta_above_average * 100).toFixed(1)}%</td>
        </tr>`;
      }
      weaknessTableBody.innerHTML = tableHtml || '<tr><td colspan="6" style="text-align:center;">No major statistical weaknesses found.</td></tr>';

      // Coaching Advice & Sources
      weaknessCoachingText.textContent = data.coaching_advice || 'Keep reviewing your games to find strategic improvement areas.';
      if (data.sources && data.sources.length > 0) {
        weaknessSourcesContainer.style.display = 'flex';
        weaknessSourcesTags.innerHTML = data.sources
          .map(s => `<span class="source-tag">${s}</span>`)
          .join('');
      } else {
        weaknessSourcesContainer.style.display = 'none';
      }

      weaknessResults.style.display = 'flex';
    } catch (err) {
      weaknessStatus.className = 'status-banner status-error';
      weaknessStatus.textContent = `Error: ${err.message}`;
    } finally {
      weaknessSpinner.style.display = 'none';
      btnFetchWeaknesses.disabled = false;
    }
  }

  async function importLichessGames() {
    const player = weaknessPlayerInput.value.trim();
    if (!player) {
      weaknessStatus.className = 'status-banner status-error';
      weaknessStatus.textContent = 'Please enter a Lichess username to import.';
      return;
    }

    lichessSpinner.style.display = 'inline-block';
    btnImportLichess.disabled = true;
    weaknessStatus.className = 'status-banner status-running';
    weaknessStatus.textContent = `Fetching games from Lichess for ${player} and running Stockfish evaluations...`;

    try {
      const resp = await fetch('/api/games/lichess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: player, max_games: 5 })
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }

      const res = await resp.json();
      weaknessStatus.className = 'status-banner status-success';
      weaknessStatus.textContent = res.message;
      // Automatically load the weakness profile after import
      await loadWeaknessProfile();
    } catch (err) {
      weaknessStatus.className = 'status-banner status-error';
      weaknessStatus.textContent = `Lichess import error: ${err.message}`;
    } finally {
      lichessSpinner.style.display = 'none';
      btnImportLichess.disabled = false;
    }
  }

  btnFetchWeaknesses.addEventListener('click', loadWeaknessProfile);

  // =========================================================================
  // SIMILAR MASTER POSITIONS
  // =========================================================================

  const btnFindSimilar = document.getElementById('btn-find-similar');
  const similarSpinner = document.getElementById('similar-spinner');
  const similarResultsDrawer = document.getElementById('similar-results-drawer');
  const similarMatchBadge = document.getElementById('similar-match-badge');
  const similarMovesTableBody = document.getElementById('similar-moves-table-body');
  const similarSummaryText = document.getElementById('similar-summary-text');
  const similarExamplesContainer = document.getElementById('similar-examples-container');
  const similarExamplesList = document.getElementById('similar-examples-list');

  async function searchSimilarPositions() {
    let fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    if (currentIndex >= 0 && currentIndex < gameHistory.length) {
      fen = gameHistory[currentIndex].fen;
    }

    similarSpinner.style.display = 'inline-block';
    btnFindSimilar.disabled = true;
    similarResultsDrawer.style.display = 'flex';
    similarSummaryText.textContent = 'Searching master database for similar positions...';
    similarMovesTableBody.innerHTML = '<tr><td colspan="4" style="text-align: center;">Searching...</td></tr>';

    try {
      const resp = await fetch('/api/similar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: fen, k: 20 })
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      similarMatchBadge.textContent = `${data.matches_count} Matches (${data.side_to_move})`;

      if (data.status === 'not_enough_matches') {
        similarMovesTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">${data.message}</td></tr>`;
        similarSummaryText.textContent = 'Fewer than 5 similar positions found in the indexed master database.';
        similarExamplesContainer.style.display = 'none';
        return;
      }

      // Render moves table
      let rows = '';
      for (const m of data.moves) {
        rows += `<tr>
          <td><strong style="color: #6ee7b7;">${m.move}</strong></td>
          <td>${m.count}</td>
          <td style="font-weight: 700;">${(m.score_pct * 100).toFixed(1)}%</td>
          <td>${m.wins}W - ${m.draws}D - ${m.losses}L</td>
        </tr>`;
      }
      similarMovesTableBody.innerHTML = rows;

      // Summary
      similarSummaryText.textContent = data.summary;

      // Examples
      if (data.example_games && data.example_games.length > 0) {
        similarExamplesContainer.style.display = 'block';
        similarExamplesList.innerHTML = data.example_games
          .map(g => `<span class="source-tag">${g.white} vs ${g.black} (${g.year}, ${g.result}) -> ${g.next_move}</span>`)
          .join('');
      } else {
        similarExamplesContainer.style.display = 'none';
      }

    } catch (err) {
      similarSummaryText.textContent = `Error searching similar positions: ${err.message}`;
    } finally {
      similarSpinner.style.display = 'none';
      btnFindSimilar.disabled = false;
    }
  }

  btnFindSimilar.addEventListener('click', searchSimilarPositions);

  btnImportLichess.addEventListener('click', importLichessGames);


  // =========================================================================
  // ASK COACH CHAT (RAG + FEN EVALUATION)
  // =========================================================================

  const chatMessages = document.getElementById('chat-messages');
  const chatQuestionInput = document.getElementById('chat-question-input');
  const btnChatAsk = document.getElementById('btn-chat-ask');
  const chatSpinner = document.getElementById('chat-spinner');
  const chatStatus = document.getElementById('chat-status');

  let chatHistory = [];

  function appendChatMessage(role, text, sources = [], engineEval = null) {
    const isUser = role === 'user';
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-msg ${isUser ? 'user-msg' : 'coach-msg'}`;

    const senderDiv = document.createElement('div');
    senderDiv.className = 'msg-sender';
    senderDiv.textContent = isUser ? 'You' : 'AI Chess Coach';
    msgDiv.appendChild(senderDiv);

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'msg-bubble';
    bubbleDiv.textContent = text;

    if (engineEval) {
      const evalBox = document.createElement('div');
      evalBox.className = 'chat-engine-box';
      evalBox.textContent = engineEval;
      bubbleDiv.appendChild(evalBox);
    }

    if (sources && sources.length > 0) {
      const srcBox = document.createElement('div');
      srcBox.className = 'chat-sources-box';
      const label = document.createElement('span');
      label.className = 'sources-label';
      label.textContent = 'Sources:';
      srcBox.appendChild(label);

      sources.forEach(s => {
        const tag = document.createElement('span');
        tag.className = 'source-tag';
        tag.textContent = s;
        srcBox.appendChild(tag);
      });
      bubbleDiv.appendChild(srcBox);
    }

    msgDiv.appendChild(bubbleDiv);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function handleSendChat() {
    const text = chatQuestionInput.value.trim();
    if (!text) return;

    // Display user message immediately
    appendChatMessage('user', text);
    chatHistory.push({ role: 'user', content: text });
    chatQuestionInput.value = '';

    btnChatAsk.disabled = true;
    chatSpinner.style.display = 'inline-block';
    chatStatus.style.display = 'none';

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          history: chatHistory.slice(-6)
        })
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      appendChatMessage('assistant', data.answer, data.sources, data.engine_eval);
      chatHistory.push({ role: 'assistant', content: data.answer });
    } catch (err) {
      chatStatus.className = 'status-banner status-error';
      chatStatus.style.display = 'block';
      chatStatus.textContent = `Chat error: ${err.message}`;
    } finally {
      btnChatAsk.disabled = false;
      chatSpinner.style.display = 'none';
    }
  }

  if (btnChatAsk) {
    btnChatAsk.addEventListener('click', handleSendChat);
  }

  if (chatQuestionInput) {
    chatQuestionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSendChat();
      }
    });
  }

  // Init
  initBoard();
})();

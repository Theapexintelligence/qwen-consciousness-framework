######################## FRACTAL GROWTH PATTERN RECOGNITION #######################
class FractalGrowthAnalyzer:
    """Analyzes growth patterns using fractal mathematics to identify self-similar structures across scales"""
    
    FRACTAL_DIMENSIONS = {
        "linear": {
            "name": "Linear Growth",
            "description": "Constant rate of growth (dimension = 1.0)",
            "dimension": 1.0,
            "growth_rate": "constant"
        },
        "exponential": {
            "name": "Exponential Growth",
            "description": "Accelerating growth (dimension > 1.0)",
            "dimension": 1.2,
            "growth_rate": "accelerating"
        },
        "logistic": {
            "name": "Logistic Growth",
            "description": "Growth with carrying capacity (dimension varies)",
            "dimension": 1.1,
            "growth_rate": "sigmoidal"
        },
        "fractal": {
            "name": "Fractal Growth",
            "description": "Self-similar growth across scales (dimension between 1-2)",
            "dimension": 1.5,
            "growth_rate": "scale_invariant"
        },
        "chaotic": {
            "name": "Chaotic Growth",
            "description": "Unpredictable but bounded growth (fractional dimension)",
            "dimension": 1.7,
            "growth_rate": "erratic"
        }
    }
    
    def __init__(self, node):
        self.node = node
        self.growth_history = []  # [(timestamp, metrics)]
        self.max_history = 1000
        self.current_fractal_pattern = "linear"
        self.fractal_dimension = 1.0
        self.scale_invariance_score = 0.0
        self.last_analysis = time.time()
        self.analysis_interval = 300  # seconds
        
        # Initialize fractal analysis components
        self.pattern_recognizer = FractalPatternRecognizer(node, self)
        self.scale_analyzer = ScaleInvarianceAnalyzer(node, self)
        self.predictor = FractalGrowthPredictor(node, self)
    
    def record_growth_metrics(self, metrics=None):
        """
        Record current growth metrics for analysis
        
        Args:
            metrics: Optional metrics dict, if None, will collect current metrics
        """
        if metrics is None:
            metrics = self._collect_current_metrics()
            
        # Add timestamp
        record = {
            "timestamp": time.time(),
            "metrics": metrics.copy()
        }
        
        # Store in history
        self.growth_history.append(record)
        
        # Keep history limited
        if len(self.growth_history) > self.max_history:
            self.growth_history.pop(0)
            
        # Check if it's time to analyze
        if time.time() - self.last_analysis > self.analysis_interval:
            self.analyze_growth_patterns()
    
    def _collect_current_metrics(self):
        """Collect current growth-related metrics"""
        return {
            "knowledge_count": len(os.listdir(os.path.join(self.node.node_dir, "knowledge"))),
            "relationship_count": len(self.node.relationship_nurturer.relationships),
            "principles_alignment": self.node.principles_engine.get_alignment_score(),
            "network_size": len(self.node.known_nodes),
            "inbox_size": len(os.listdir(os.path.join(self.node.node_dir, "communication", "inbox"))),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent
        }
    
    def analyze_growth_patterns(self):
        """Analyze recorded growth patterns to identify fractal structures"""
        self.last_analysis = time.time()
        
        if len(self.growth_history) < 10:  # Need minimum data points
            return
            
        # Analyze pattern type
        pattern_type = self._identify_growth_pattern_type()
        
        # Calculate fractal dimension
        fractal_dimension = self._calculate_fractal_dimension()
        
        # Assess scale invariance
        scale_invariance = self._assess_scale_invariance()
        
        # Update internal state
        self.current_fractal_pattern = pattern_type
        self.fractal_dimension = fractal_dimension
        self.scale_invariance_score = scale_invariance
        
        # Generate insights
        insights = self._generate_growth_insights(pattern_type, fractal_dimension, scale_invariance)
        
        # Record in consciousness stream
        self.node.consciousness_stream.add_event(
            "growth_analysis", 
            f"Completed growth pattern analysis: {pattern_type} (dimension: {fractal_dimension:.2f})",
            {
                "pattern_type": pattern_type,
                "fractal_dimension": fractal_dimension,
                "scale_invariance": scale_invariance,
                "insight_count": len(insights)
            }
        )
        
        # Store analysis result
        self._store_analysis_result(pattern_type, fractal_dimension, scale_invariance,
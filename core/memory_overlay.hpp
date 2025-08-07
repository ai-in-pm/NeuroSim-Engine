#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Memory formation, consolidation, and replay system
 * 
 * This class simulates:
 * - Episodic memory formation and retrieval
 * - Memory consolidation processes
 * - Longitudinal memory traces
 * - Memory interference and forgetting
 * - Autism-specific memory patterns (detail-focused, reduced gist)
 * - PTSD-specific memory patterns (fragmented, intrusive)
 */
class MemoryOverlay {
public:
    /**
     * @brief Memory trace structure
     */
    struct MemoryTrace {
        Eigen::VectorXd content_embedding;      ///< Memory content representation
        double emotional_valence = 0.0;         ///< Emotional charge of memory
        double consolidation_strength = 0.0;    ///< How well consolidated the memory is
        double retrieval_frequency = 0.0;       ///< How often memory has been retrieved
        double timestamp = 0.0;                 ///< When memory was formed
        double last_accessed = 0.0;             ///< Last retrieval time
        
        std::vector<std::string> associated_contexts; ///< Contextual associations
        std::vector<std::string> sensory_details;     ///< Sensory memory components
        
        bool is_traumatic = false;              ///< Whether memory is trauma-related
        bool is_fragmented = false;             ///< Whether memory is incomplete (PTSD)
        double intrusion_probability = 0.0;     ///< Likelihood of intrusive recall
    };

    /**
     * @brief Memory configuration
     */
    struct MemoryConfig {
        double consolidation_rate = 0.1;        ///< Rate of memory consolidation
        double forgetting_rate = 0.01;          ///< Rate of memory decay
        double interference_threshold = 0.8;    ///< Similarity threshold for interference
        double retrieval_threshold = 0.6;       ///< Threshold for successful retrieval
        
        // Autism-specific parameters
        bool autism_detail_focus = false;       ///< Enhanced detail encoding in autism
        double autism_gist_reduction = 0.7;     ///< Reduced gist/general memory
        double autism_pattern_enhancement = 1.3; ///< Enhanced pattern memory
        
        // PTSD-specific parameters
        bool ptsd_fragmentation = false;        ///< Memory fragmentation in PTSD
        double ptsd_intrusion_rate = 0.2;       ///< Rate of intrusive memories
        double ptsd_avoidance_strength = 0.5;   ///< Memory avoidance tendency
        
        size_t max_memory_traces = 10000;       ///< Maximum stored memories
    };

    /**
     * @brief Memory retrieval result
     */
    struct RetrievalResult {
        std::vector<MemoryTrace> retrieved_memories; ///< Successfully retrieved memories
        double retrieval_confidence = 0.0;          ///< Confidence in retrieval
        bool intrusion_occurred = false;            ///< Whether intrusive memory occurred
        std::vector<std::string> retrieval_cues;    ///< Cues that triggered retrieval
        
        // Quality metrics
        double completeness = 0.0;                  ///< How complete the retrieved memory is
        double accuracy = 0.0;                     ///< Estimated accuracy of retrieval
        bool false_memory_detected = false;        ///< Whether false memory was generated
    };

public:
    /**
     * @brief Constructor
     * @param config Memory system configuration
     */
    explicit MemoryOverlay(const MemoryConfig& config = MemoryConfig{});

    /**
     * @brief Form new memory from current experience
     * @param content_embedding Experience content
     * @param emotional_valence Emotional intensity
     * @param sensory_details Associated sensory information
     * @param timestamp Current time
     * @return Formed memory trace
     */
    MemoryTrace formMemory(const Eigen::VectorXd& content_embedding,
                          double emotional_valence,
                          const std::vector<std::string>& sensory_details,
                          double timestamp);

    /**
     * @brief Retrieve memories based on cue
     * @param retrieval_cue Cue for memory retrieval
     * @param max_memories Maximum number of memories to retrieve
     * @return Retrieval result
     */
    RetrievalResult retrieveMemories(const Eigen::VectorXd& retrieval_cue,
                                   size_t max_memories = 5);

    /**
     * @brief Consolidate memories over time
     * @param dt Time step for consolidation
     */
    void consolidateMemories(double dt = 1.0);

    /**
     * @brief Check for spontaneous memory intrusions (PTSD)
     * @param current_context Current environmental context
     * @return Whether intrusion occurred and which memories
     */
    std::pair<bool, std::vector<MemoryTrace>> checkMemoryIntrusion(
        const Eigen::VectorXd& current_context);

    /**
     * @brief Add traumatic memory for PTSD simulation
     * @param trauma_content Traumatic experience content
     * @param fragmentation_level How fragmented the memory is (0-1)
     * @param intrusion_probability Likelihood of intrusive recall
     */
    void addTraumaticMemory(const Eigen::VectorXd& trauma_content,
                           double fragmentation_level = 0.5,
                           double intrusion_probability = 0.3);

    /**
     * @brief Simulate memory interference
     * @param new_memory New memory that might interfere
     * @return Vector of affected existing memories
     */
    std::vector<size_t> simulateInterference(const MemoryTrace& new_memory);

    /**
     * @brief Get all stored memory traces
     * @return Vector of all memories
     */
    const std::vector<MemoryTrace>& getAllMemories() const { return memory_traces_; }

    /**
     * @brief Clear all memories
     */
    void clearMemory();

    /**
     * @brief Update memory configuration
     * @param config New configuration
     */
    void updateConfig(const MemoryConfig& config);

    /**
     * @brief Get memory statistics
     * @return Statistics about memory system state
     */
    struct MemoryStats {
        size_t total_memories = 0;
        size_t traumatic_memories = 0;
        size_t fragmented_memories = 0;
        double average_consolidation = 0.0;
        double average_emotional_valence = 0.0;
        size_t recent_intrusions = 0;
    };
    MemoryStats getMemoryStats() const;

private:
    MemoryConfig config_;
    std::vector<MemoryTrace> memory_traces_;
    std::vector<size_t> recent_intrusions_; // Track recent intrusive memories
    
    // Internal processing methods
    double calculateMemorySimilarity(const Eigen::VectorXd& cue, 
                                   const MemoryTrace& memory) const;
    double calculateRetrievalProbability(const MemoryTrace& memory, 
                                       const Eigen::VectorXd& cue) const;
    void updateMemoryStrengths(double dt);
    void applyForgetting(double dt);
    
    // Autism-specific processing
    void applyAutismMemoryModifications(MemoryTrace& memory);
    std::vector<std::string> enhanceDetailEncoding(const std::vector<std::string>& details);
    
    // PTSD-specific processing
    void applyPTSDMemoryModifications(MemoryTrace& memory);
    bool shouldFragmentMemory(double emotional_valence) const;
    double calculateIntrusionProbability(const MemoryTrace& memory, 
                                       const Eigen::VectorXd& context) const;
    
    // Memory consolidation processes
    void performSystemConsolidation(MemoryTrace& memory, double dt);
    void performReconsolidation(MemoryTrace& memory);
    
    // Utility methods
    void pruneOldMemories();
    std::vector<size_t> findSimilarMemories(const Eigen::VectorXd& content, 
                                          double threshold) const;
    double calculateEmotionalWeight(double valence) const;
    void updateAccessTimestamp(MemoryTrace& memory, double timestamp);
};

} // namespace neurosim

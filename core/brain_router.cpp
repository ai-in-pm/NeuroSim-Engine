#include "brain_router.hpp"
#include <algorithm>
#include <random>
#include <unordered_set>

namespace neurosim {

// Static lexicon data (simplified for initial implementation)
const std::unordered_map<std::string, double> BrainRouter::emotional_lexicon_ = {
    {"happy", 0.8}, {"sad", -0.7}, {"angry", -0.6}, {"fear", -0.9}, {"joy", 0.9},
    {"scared", -0.8}, {"worried", -0.5}, {"excited", 0.7}, {"calm", 0.3},
    {"anxious", -0.6}, {"love", 0.9}, {"hate", -0.8}, {"good", 0.5}, {"bad", -0.5}
};

const std::unordered_map<std::string, double> BrainRouter::threat_lexicon_ = {
    {"danger", 0.9}, {"safe", -0.5}, {"threat", 0.8}, {"attack", 0.9}, {"protect", -0.3},
    {"explosion", 0.95}, {"gun", 0.8}, {"weapon", 0.7}, {"enemy", 0.8}, {"combat", 0.9},
    {"loud", 0.4}, {"noise", 0.3}, {"unknown", 0.4}, {"stranger", 0.5}, {"dark", 0.3}
};

const std::unordered_map<std::string, double> BrainRouter::social_lexicon_ = {
    {"person", 0.7}, {"people", 0.8}, {"friend", 0.6}, {"family", 0.5}, {"stranger", 0.8},
    {"crowd", 0.9}, {"alone", 0.4}, {"together", 0.6}, {"talk", 0.5}, {"speak", 0.5},
    {"eye", 0.7}, {"contact", 0.6}, {"social", 0.8}, {"interaction", 0.7}
};

const std::unordered_map<std::string, std::vector<std::string>> BrainRouter::semantic_categories_ = {
    {"emotion", {"happy", "sad", "angry", "fear", "joy", "scared", "worried", "excited", "calm", "anxious", "love", "hate"}},
    {"threat", {"danger", "threat", "attack", "explosion", "gun", "weapon", "enemy", "combat", "loud", "noise"}},
    {"social", {"person", "people", "friend", "family", "stranger", "crowd", "talk", "speak", "eye", "contact", "social", "interaction"}},
    {"sensory", {"loud", "bright", "dark", "noise", "sound", "light", "touch", "feel", "see", "hear"}},
    {"body", {"pain", "hurt", "tired", "sick", "healthy", "strong", "weak", "heart", "breath", "body"}}
};

BrainRouter::BrainRouter(const RoutingConfig& config) : config_(config) {
}

std::vector<BrainRouter::RegionActivation> BrainRouter::routeTokens(
    const std::vector<std::string>& tokens,
    const Eigen::VectorXd& multimodal_context) {
    
    std::vector<RegionActivation> activations;
    
    // Analyze all tokens
    std::vector<TokenAnalysis> token_analyses;
    for (const auto& token : tokens) {
        token_analyses.push_back(analyzeToken(token));
    }
    
    // Route to specific brain regions
    activations.push_back(routeToAmygdala(token_analyses));
    activations.push_back(routeToHippocampus(token_analyses));
    activations.push_back(routeToInsula(token_analyses));
    activations.push_back(routeToPrefrontal(token_analyses));
    activations.push_back(routeToCerebellum(token_analyses));
    activations.push_back(routeToSTG(token_analyses));
    activations.push_back(routeToACC(token_analyses));
    
    // Apply autism modifications
    if (config_.autism_hypersensitivity) {
        applyAutismModifications(activations);
    }
    
    // Apply PTSD modifications
    if (config_.ptsd_hypervigilance) {
        applyPTSDModifications(activations);
    }
    
    // Store in history
    activation_history_.push_back(activations);
    if (activation_history_.size() > 1000) {
        activation_history_.erase(activation_history_.begin());
    }
    
    return activations;
}

BrainRouter::TokenAnalysis BrainRouter::analyzeToken(const std::string& token) const {
    TokenAnalysis analysis;
    analysis.token = token;
    
    analysis.emotional_valence = calculateEmotionalValence(token);
    analysis.arousal_level = calculateArousalLevel(token);
    analysis.social_relevance = calculateSocialRelevance(token);
    analysis.threat_level = calculateThreatLevel(token);
    analysis.sensory_intensity = calculateSensoryIntensity(token);
    analysis.semantic_categories = classifySemantics(token);
    
    return analysis;
}

double BrainRouter::calculateEmotionalValence(const std::string& token) const {
    auto it = emotional_lexicon_.find(token);
    return (it != emotional_lexicon_.end()) ? it->second : 0.0;
}

double BrainRouter::calculateArousalLevel(const std::string& token) const {
    // High arousal for emotional and threat words
    double emotional_magnitude = std::abs(calculateEmotionalValence(token));
    double threat_level = calculateThreatLevel(token);
    return std::min(1.0, emotional_magnitude + threat_level);
}

double BrainRouter::calculateSocialRelevance(const std::string& token) const {
    auto it = social_lexicon_.find(token);
    return (it != social_lexicon_.end()) ? it->second : 0.0;
}

double BrainRouter::calculateThreatLevel(const std::string& token) const {
    auto it = threat_lexicon_.find(token);
    return (it != threat_lexicon_.end()) ? it->second : 0.0;
}

double BrainRouter::calculateSensoryIntensity(const std::string& token) const {
    // Simple heuristic based on word characteristics
    if (token.find("loud") != std::string::npos || 
        token.find("bright") != std::string::npos ||
        token.find("noise") != std::string::npos) {
        return 0.8;
    }
    return 0.2;
}

std::vector<std::string> BrainRouter::classifySemantics(const std::string& token) const {
    std::vector<std::string> categories;
    
    for (const auto& [category, words] : semantic_categories_) {
        if (std::find(words.begin(), words.end(), token) != words.end()) {
            categories.push_back(category);
        }
    }
    
    return categories;
}

BrainRouter::RegionActivation BrainRouter::routeToAmygdala(const std::vector<TokenAnalysis>& tokens) const {
    RegionActivation activation;
    activation.region_name = "Amygdala";
    
    double total_threat = 0.0;
    double total_emotional = 0.0;
    std::vector<std::string> contributing_tokens;
    
    for (const auto& token : tokens) {
        if (token.threat_level > 0.3 || std::abs(token.emotional_valence) > 0.5) {
            total_threat += token.threat_level;
            total_emotional += std::abs(token.emotional_valence);
            contributing_tokens.push_back(token.token);
        }
    }
    
    activation.activation_strength = std::min(1.0, (total_threat + total_emotional) * config_.amygdala_sensitivity);
    activation.latency_ms = calculateLatency("Amygdala", activation.activation_strength);
    activation.contributing_tokens = contributing_tokens;
    activation.activation_reason = generateActivationReason("Amygdala", tokens);
    
    return activation;
}

BrainRouter::RegionActivation BrainRouter::routeToHippocampus(const std::vector<TokenAnalysis>& tokens) const {
    RegionActivation activation;
    activation.region_name = "Hippocampus";
    
    // Hippocampus activates for memory-related and contextual processing
    double memory_relevance = 0.0;
    std::vector<std::string> contributing_tokens;
    
    for (const auto& token : tokens) {
        // Simple heuristic: any meaningful content activates hippocampus
        if (!token.semantic_categories.empty()) {
            memory_relevance += 0.3;
            contributing_tokens.push_back(token.token);
        }
    }
    
    activation.activation_strength = std::min(1.0, memory_relevance);
    activation.latency_ms = calculateLatency("Hippocampus", activation.activation_strength);
    activation.contributing_tokens = contributing_tokens;
    activation.activation_reason = "Memory encoding and contextual processing";
    
    return activation;
}

BrainRouter::RegionActivation BrainRouter::routeToInsula(const std::vector<TokenAnalysis>& tokens) const {
    RegionActivation activation;
    activation.region_name = "Insula";
    
    double interoceptive_relevance = 0.0;
    std::vector<std::string> contributing_tokens;
    
    for (const auto& token : tokens) {
        if (token.sensory_intensity > 0.4 || std::abs(token.emotional_valence) > 0.4) {
            interoceptive_relevance += token.sensory_intensity + std::abs(token.emotional_valence) * 0.5;
            contributing_tokens.push_back(token.token);
        }
    }
    
    activation.activation_strength = std::min(1.0, interoceptive_relevance);
    activation.latency_ms = calculateLatency("Insula", activation.activation_strength);
    activation.contributing_tokens = contributing_tokens;
    activation.activation_reason = "Interoceptive and emotional processing";
    
    return activation;
}

BrainRouter::RegionActivation BrainRouter::routeToPrefrontal(const std::vector<TokenAnalysis>& tokens) const {
    RegionActivation activation;
    activation.region_name = "PFC";
    
    // PFC activates for cognitive control and inhibition
    double cognitive_load = std::min(1.0, static_cast<double>(tokens.size()) * 0.2);
    
    activation.activation_strength = cognitive_load * config_.prefrontal_inhibition;
    activation.latency_ms = calculateLatency("PFC", activation.activation_strength);
    activation.activation_reason = "Executive control and cognitive processing";
    
    return activation;
}

BrainRouter::RegionActivation BrainRouter::routeToCerebellum(const std::vector<TokenAnalysis>& tokens) const {
    RegionActivation activation;
    activation.region_name = "Cerebellum";
    
    // Cerebellum activates for coordination and timing
    double coordination_demand = std::min(1.0, static_cast<double>(tokens.size()) * 0.15);
    
    activation.activation_strength = coordination_demand;
    activation.latency_ms = calculateLatency("Cerebellum", activation.activation_strength);
    activation.activation_reason = "Motor and cognitive coordination";
    
    return activation;
}

BrainRouter::RegionActivation BrainRouter::routeToSTG(const std::vector<TokenAnalysis>& tokens) const {
    RegionActivation activation;
    activation.region_name = "STG";
    
    // STG activates for auditory and language processing
    double language_processing = std::min(1.0, static_cast<double>(tokens.size()) * 0.25);
    
    activation.activation_strength = language_processing;
    activation.latency_ms = calculateLatency("STG", activation.activation_strength);
    activation.activation_reason = "Auditory and language processing";
    
    return activation;
}

BrainRouter::RegionActivation BrainRouter::routeToACC(const std::vector<TokenAnalysis>& tokens) const {
    RegionActivation activation;
    activation.region_name = "ACC";
    
    double conflict_monitoring = 0.0;
    for (const auto& token : tokens) {
        if (std::abs(token.emotional_valence) > 0.5 || token.threat_level > 0.4) {
            conflict_monitoring += 0.3;
        }
    }
    
    activation.activation_strength = std::min(1.0, conflict_monitoring);
    activation.latency_ms = calculateLatency("ACC", activation.activation_strength);
    activation.activation_reason = "Conflict monitoring and emotional regulation";
    
    return activation;
}

void BrainRouter::applyAutismModifications(std::vector<RegionActivation>& activations) const {
    for (auto& activation : activations) {
        if (activation.region_name == "Amygdala") {
            // Enhanced social threat detection
            activation.activation_strength *= 1.3;
        } else if (activation.region_name == "Insula") {
            // Sensory hypersensitivity
            activation.activation_strength *= 1.4;
        } else if (activation.region_name == "PFC") {
            // Reduced inhibitory control
            activation.activation_strength *= 0.7;
        }
    }
}

void BrainRouter::applyPTSDModifications(std::vector<RegionActivation>& activations) const {
    for (auto& activation : activations) {
        if (activation.region_name == "Amygdala") {
            // Hypervigilance and threat sensitivity
            activation.activation_strength *= 1.5;
            activation.latency_ms *= 0.7; // Faster threat detection
        } else if (activation.region_name == "PFC") {
            // Impaired inhibitory control
            activation.activation_strength *= 0.6;
        } else if (activation.region_name == "Hippocampus") {
            // Memory fragmentation
            activation.activation_strength *= 0.8;
        }
    }
}

double BrainRouter::calculateLatency(const std::string& region_name, double activation_strength) const {
    // Base latencies (in milliseconds)
    std::unordered_map<std::string, double> base_latencies = {
        {"Amygdala", 100.0}, {"Hippocampus", 150.0}, {"Insula", 120.0},
        {"PFC", 200.0}, {"Cerebellum", 80.0}, {"STG", 110.0}, {"ACC", 130.0}
    };
    
    double base_latency = base_latencies.count(region_name) ? base_latencies[region_name] : 150.0;
    
    // Higher activation = faster response
    return base_latency * (1.0 - activation_strength * 0.3);
}

std::string BrainRouter::generateActivationReason(const std::string& region_name, 
                                                 const std::vector<TokenAnalysis>& tokens) const {
    if (region_name == "Amygdala") {
        return "Threat detection and emotional processing";
    } else if (region_name == "Hippocampus") {
        return "Memory formation and contextual processing";
    } else if (region_name == "Insula") {
        return "Interoceptive and emotional awareness";
    } else if (region_name == "PFC") {
        return "Executive control and cognitive regulation";
    } else if (region_name == "Cerebellum") {
        return "Motor and cognitive coordination";
    } else if (region_name == "STG") {
        return "Auditory and language processing";
    } else if (region_name == "ACC") {
        return "Conflict monitoring and emotional regulation";
    }
    return "General neural processing";
}

void BrainRouter::updateConfig(const RoutingConfig& config) {
    config_ = config;
}

void BrainRouter::registerBrainRegion(const std::string& region_name, std::shared_ptr<BrainRegion> region) {
    brain_regions_[region_name] = region;
}

std::vector<std::vector<BrainRouter::RegionActivation>> BrainRouter::getActivationHistory() const {
    return activation_history_;
}

void BrainRouter::clearHistory() {
    activation_history_.clear();
}

} // namespace neurosim

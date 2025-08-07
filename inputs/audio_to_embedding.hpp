#pragma once

#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>

namespace neurosim {

/**
 * @brief Auditory embedding generator for audio processing
 * 
 * This class simulates auditory feature extraction, converting audio input
 * into embeddings for neural simulation. It includes:
 * - Spectral feature extraction (pitch, volume, timbre)
 * - Speech recognition and emotional prosody
 * - Environmental sound classification
 * - Music and rhythm processing
 * - Autism-specific auditory processing differences
 * - PTSD-specific threat detection in auditory stimuli
 */
class AudioToEmbedding {
public:
    /**
     * @brief Auditory processing configuration
     */
    struct AudioConfig {
        size_t embedding_dimension = 256;        ///< Output embedding size
        double sample_rate = 44100.0;            ///< Audio sample rate (Hz)
        size_t window_size = 1024;               ///< FFT window size
        size_t hop_length = 512;                 ///< Hop length for STFT
        bool enable_speech_processing = true;    ///< Enable speech recognition
        bool enable_emotion_detection = true;    ///< Enable emotional prosody
        bool enable_music_processing = true;     ///< Enable music analysis
        
        // Autism-specific parameters
        bool autism_auditory_hypersensitivity = false; ///< Enhanced auditory sensitivity
        double autism_frequency_selectivity = 1.3;     ///< Enhanced frequency discrimination
        double autism_volume_sensitivity = 1.5;        ///< Enhanced volume sensitivity
        bool autism_speech_processing_differences = false; ///< Altered speech processing
        double autism_background_noise_difficulty = 1.4;   ///< Difficulty filtering background noise
        
        // PTSD-specific parameters
        bool ptsd_hypervigilance = false;        ///< Enhanced threat detection
        double ptsd_startle_threshold = 0.3;     ///< Lower threshold for startle
        std::vector<std::string> ptsd_trigger_sounds; ///< Sounds that trigger PTSD
        bool ptsd_combat_audio_sensitivity = false;   ///< Combat-specific audio triggers
        double ptsd_hyperacusis = 1.2;           ///< Sound sensitivity in PTSD
    };

    /**
     * @brief Audio input data structure
     */
    struct AudioInput {
        std::vector<double> audio_data;          ///< Raw audio samples
        double sample_rate = 44100.0;            ///< Sample rate (Hz)
        size_t num_channels = 1;                 ///< Number of audio channels
        double duration_seconds = 0.0;           ///< Audio duration
        double timestamp = 0.0;                  ///< Input timestamp
        double rms_volume = 0.0;                 ///< RMS volume level
        std::string audio_source = "microphone"; ///< Audio source type
    };

    /**
     * @brief Auditory processing result
     */
    struct AudioEmbedding {
        Eigen::VectorXd feature_embedding;       ///< Main auditory feature vector
        
        // Spectral features
        std::vector<double> pitch_contour;       ///< Fundamental frequency over time
        double average_pitch = 0.0;              ///< Average pitch (Hz)
        double pitch_variance = 0.0;             ///< Pitch variability
        double volume_level = 0.0;               ///< Average volume (dB)
        std::vector<double> spectral_centroid;   ///< Spectral centroid over time
        std::vector<double> mfcc_features;       ///< Mel-frequency cepstral coefficients
        
        // High-level features
        std::string speech_content;              ///< Recognized speech (if any)
        std::string emotional_tone;              ///< Detected emotional prosody
        std::string sound_category;              ///< Environmental sound category
        double speech_clarity = 0.0;             ///< Speech intelligibility
        bool music_detected = false;             ///< Whether music is present
        
        // Temporal features
        double onset_density = 0.0;              ///< Rate of sound onsets
        double rhythm_strength = 0.0;            ///< Rhythmic regularity
        std::vector<double> tempo_estimates;     ///< Estimated tempo(s)
        
        // Autism-specific metrics
        struct {
            double hypersensitivity_activation = 0.0; ///< Auditory hypersensitivity level
            double frequency_discrimination = 0.0;     ///< Frequency processing quality
            double background_noise_interference = 0.0; ///< Background noise impact
            bool sensory_overload = false;             ///< Auditory sensory overload
            std::vector<std::string> overwhelming_frequencies; ///< Problematic frequency ranges
        } autism_metrics;
        
        // PTSD-specific metrics
        struct {
            double threat_level = 0.0;            ///< Detected threat level
            std::vector<std::string> threat_sounds; ///< Threatening sounds detected
            bool startle_trigger = false;         ///< Whether auditory startle occurred
            double hypervigilance_activation = 0.0; ///< Hypervigilance level
            bool combat_trigger = false;          ///< Combat-related trigger detected
        } ptsd_metrics;
        
        double processing_confidence = 1.0;       ///< Confidence in processing result
        double processing_time_ms = 0.0;          ///< Simulated processing time
    };

public:
    /**
     * @brief Constructor
     * @param config Audio processing configuration
     */
    explicit AudioToEmbedding(const AudioConfig& config = AudioConfig{});

    /**
     * @brief Process audio input and generate embedding
     * @param input Audio input data
     * @return Audio embedding result
     */
    AudioEmbedding processAudio(const AudioInput& input);

    /**
     * @brief Process audio from file path (convenience method)
     * @param audio_path Path to audio file
     * @return Audio embedding result
     */
    AudioEmbedding processAudioFile(const std::string& audio_path);

    /**
     * @brief Process simulated audio scene
     * @param audio_description Text description of audio scene
     * @return Simulated audio embedding
     */
    AudioEmbedding processSimulatedAudio(const std::string& audio_description);

    /**
     * @brief Update processing configuration
     * @param config New configuration
     */
    void updateConfig(const AudioConfig& config);

    /**
     * @brief Get current configuration
     * @return Current audio config
     */
    const AudioConfig& getConfig() const { return config_; }

    /**
     * @brief Add PTSD trigger sound for threat detection
     * @param sound_name Sound that triggers PTSD response
     * @param threat_level Threat level (0-1)
     */
    void addPTSDTriggerSound(const std::string& sound_name, double threat_level = 0.8);

    /**
     * @brief Add combat-specific audio triggers
     * @param combat_sounds Vector of combat-related sounds
     */
    void addCombatTriggers(const std::vector<std::string>& combat_sounds);

    /**
     * @brief Get processing history for analysis
     * @return Vector of recent audio processing results
     */
    std::vector<AudioEmbedding> getProcessingHistory() const;

    /**
     * @brief Clear processing history
     */
    void clearHistory();

private:
    AudioConfig config_;
    std::vector<AudioEmbedding> processing_history_;
    
    // Core audio processing methods
    Eigen::VectorXd extractSpectralFeatures(const AudioInput& input);
    std::vector<double> extractPitchContour(const AudioInput& input);
    std::vector<double> extractMFCC(const AudioInput& input);
    double calculateVolume(const AudioInput& input);
    std::vector<double> calculateSpectralCentroid(const AudioInput& input);
    
    // High-level processing
    std::string recognizeSpeech(const AudioInput& input);
    std::string detectEmotionalTone(const AudioInput& input);
    std::string classifyEnvironmentalSound(const AudioInput& input);
    bool detectMusic(const AudioInput& input);
    double calculateSpeechClarity(const AudioInput& input);
    
    // Temporal analysis
    double calculateOnsetDensity(const AudioInput& input);
    double calculateRhythmStrength(const AudioInput& input);
    std::vector<double> estimateTempo(const AudioInput& input);
    
    // Autism-specific processing
    void applyAutismAudioProcessing(AudioEmbedding& result, const AudioInput& input);
    double calculateHypersensitivityActivation(const AudioInput& input);
    double assessFrequencyDiscrimination(const AudioInput& input);
    double calculateBackgroundNoiseInterference(const AudioInput& input);
    bool checkSensoryOverload(const AudioInput& input);
    std::vector<std::string> identifyOverwhelmingFrequencies(const AudioInput& input);
    
    // PTSD-specific processing
    void applyPTSDAudioProcessing(AudioEmbedding& result, const AudioInput& input);
    double calculateThreatLevel(const AudioInput& input);
    std::vector<std::string> identifyThreatSounds(const AudioInput& input);
    bool checkStartleTrigger(const AudioInput& input);
    double calculateHypervigilanceActivation(const AudioInput& input);
    bool checkCombatTrigger(const AudioInput& input);
    
    // Utility methods
    double calculateProcessingTime(const AudioInput& input);
    double calculateProcessingConfidence(const AudioEmbedding& result);
    Eigen::VectorXd normalizeFeatures(const Eigen::VectorXd& features);
    std::vector<double> applyWindow(const std::vector<double>& signal, size_t start, size_t length);
    std::vector<double> computeFFT(const std::vector<double>& signal);
    
    // Simulated audio processing (for when no actual audio processing is available)
    AudioEmbedding simulateAudioProcessing(const std::string& description);
    std::vector<std::string> parseAudioDescription(const std::string& description);
    double estimateVolumeFromDescription(const std::string& description);
    std::string inferEmotionalTone(const std::string& description);
    
    // Static data for audio classification
    static const std::vector<std::string> environmental_sounds_;
    static const std::vector<std::string> emotional_tones_;
    static const std::vector<std::string> speech_keywords_;
    static const std::vector<std::string> combat_sounds_;
    static const std::vector<std::string> threat_sounds_;
};

} // namespace neurosim

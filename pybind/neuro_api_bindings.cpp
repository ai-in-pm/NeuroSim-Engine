#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "../core/simulator.hpp"
#include "../core/brain_router.hpp"
#include "../core/multimodal_fusion.hpp"
#include "../regions/amygdala.hpp"
#include "../inputs/image_to_embedding.hpp"
#include "../inputs/audio_to_embedding.hpp"
#include "../inputs/vestibular_synth.hpp"
#include "../inputs/interoceptive_sim.hpp"

namespace py = pybind11;
using namespace neurosim;

PYBIND11_MODULE(neurosim_py, m) {
    m.doc() = "NeuroSim Engine - Neural Simulation for Autism and PTSD Modeling";

    // NeuroSimulator main class
    py::class_<NeuroSimulator>(m, "NeuroSimulator")
        .def(py::init<const NeuroSimulator::Config&>(), py::arg("config") = NeuroSimulator::Config{})
        .def("process", &NeuroSimulator::process, "Process multi-modal input")
        .def("process_text", &NeuroSimulator::processText, "Process text-only input")
        .def("export_to_json", &NeuroSimulator::exportToJson, "Export state to JSON")
        .def("get_memory_traces", &NeuroSimulator::getMemoryTraces, "Get memory traces")
        .def("clear_memory", &NeuroSimulator::clearMemory, "Clear all memory")
        .def("add_trauma_memory", &NeuroSimulator::addTraumaMemory, "Add trauma memory")
        .def("update_config", &NeuroSimulator::updateConfig, "Update configuration")
        .def("get_config", &NeuroSimulator::getConfig, "Get current configuration")
        .def("reset", &NeuroSimulator::reset, "Reset simulation");

    // NeuroSimulator::Config
    py::class_<NeuroSimulator::Config>(m, "SimulatorConfig")
        .def(py::init<>())
        .def_readwrite("autism_mode", &NeuroSimulator::Config::autism_mode)
        .def_readwrite("ptsd_overlay", &NeuroSimulator::Config::ptsd_overlay)
        .def_readwrite("excitation_ratio", &NeuroSimulator::Config::excitation_ratio)
        .def_readwrite("inhibition_delay", &NeuroSimulator::Config::inhibition_delay)
        .def_readwrite("memory_threshold", &NeuroSimulator::Config::memory_threshold)
        .def_readwrite("flashback_sensitivity", &NeuroSimulator::Config::flashback_sensitivity)
        .def_readwrite("log_level", &NeuroSimulator::Config::log_level);

    // NeuroSimulator::SimulationState
    py::class_<NeuroSimulator::SimulationState>(m, "SimulationState")
        .def(py::init<>())
        .def_readwrite("response_text", &NeuroSimulator::SimulationState::response_text)
        .def_readwrite("region_activations", &NeuroSimulator::SimulationState::region_activations)
        .def_readwrite("timestamp", &NeuroSimulator::SimulationState::timestamp)
        .def_readwrite("flashback_triggered", &NeuroSimulator::SimulationState::flashback_triggered)
        .def_readwrite("active_memories", &NeuroSimulator::SimulationState::active_memories);

    // NeuroSimulator::MultiModalInput
    py::class_<NeuroSimulator::MultiModalInput>(m, "MultiModalInput")
        .def(py::init<>())
        .def_readwrite("visual_embedding", &NeuroSimulator::MultiModalInput::visual_embedding)
        .def_readwrite("audio_embedding", &NeuroSimulator::MultiModalInput::audio_embedding)
        .def_readwrite("vestibular_embedding", &NeuroSimulator::MultiModalInput::vestibular_embedding)
        .def_readwrite("interoceptive_embedding", &NeuroSimulator::MultiModalInput::interoceptive_embedding)
        .def_readwrite("text_tokens", &NeuroSimulator::MultiModalInput::text_tokens)
        .def_readwrite("timestamp", &NeuroSimulator::MultiModalInput::timestamp);

    // BrainRouter
    py::class_<BrainRouter>(m, "BrainRouter")
        .def(py::init<const BrainRouter::RoutingConfig&>(), py::arg("config") = BrainRouter::RoutingConfig{})
        .def("route_tokens", &BrainRouter::routeTokens, "Route tokens to brain regions")
        .def("analyze_token", &BrainRouter::analyzeToken, "Analyze individual token")
        .def("update_config", &BrainRouter::updateConfig, "Update routing configuration")
        .def("get_activation_history", &BrainRouter::getActivationHistory, "Get activation history")
        .def("clear_history", &BrainRouter::clearHistory, "Clear activation history");

    // BrainRouter::RoutingConfig
    py::class_<BrainRouter::RoutingConfig>(m, "RoutingConfig")
        .def(py::init<>())
        .def_readwrite("autism_hypersensitivity", &BrainRouter::RoutingConfig::autism_hypersensitivity)
        .def_readwrite("ptsd_hypervigilance", &BrainRouter::RoutingConfig::ptsd_hypervigilance)
        .def_readwrite("amygdala_sensitivity", &BrainRouter::RoutingConfig::amygdala_sensitivity)
        .def_readwrite("prefrontal_inhibition", &BrainRouter::RoutingConfig::prefrontal_inhibition)
        .def_readwrite("social_processing_bias", &BrainRouter::RoutingConfig::social_processing_bias);

    // BrainRouter::RegionActivation
    py::class_<BrainRouter::RegionActivation>(m, "RegionActivation")
        .def(py::init<>())
        .def_readwrite("region_name", &BrainRouter::RegionActivation::region_name)
        .def_readwrite("activation_strength", &BrainRouter::RegionActivation::activation_strength)
        .def_readwrite("latency_ms", &BrainRouter::RegionActivation::latency_ms)
        .def_readwrite("contributing_tokens", &BrainRouter::RegionActivation::contributing_tokens)
        .def_readwrite("activation_reason", &BrainRouter::RegionActivation::activation_reason);

    // MultiModalFusion
    py::class_<MultiModalFusion>(m, "MultiModalFusion")
        .def(py::init<const MultiModalFusion::FusionConfig&>(), py::arg("config") = MultiModalFusion::FusionConfig{})
        .def("fuse", &MultiModalFusion::fuse, "Fuse multi-modal sensory inputs")
        .def("fuse_temporal_sequence", &MultiModalFusion::fuseTemporalSequence, "Fuse temporal sequence")
        .def("update_config", &MultiModalFusion::updateConfig, "Update fusion configuration")
        .def("get_fusion_history", &MultiModalFusion::getFusionHistory, "Get fusion history")
        .def("clear_history", &MultiModalFusion::clearHistory, "Clear fusion history");

    // MultiModalFusion::FusionConfig
    py::class_<MultiModalFusion::FusionConfig>(m, "FusionConfig")
        .def(py::init<>())
        .def_readwrite("visual_weight", &MultiModalFusion::FusionConfig::visual_weight)
        .def_readwrite("auditory_weight", &MultiModalFusion::FusionConfig::auditory_weight)
        .def_readwrite("vestibular_weight", &MultiModalFusion::FusionConfig::vestibular_weight)
        .def_readwrite("interoceptive_weight", &MultiModalFusion::FusionConfig::interoceptive_weight)
        .def_readwrite("autism_sensory_hypersensitivity", &MultiModalFusion::FusionConfig::autism_sensory_hypersensitivity)
        .def_readwrite("ptsd_hypervigilance", &MultiModalFusion::FusionConfig::ptsd_hypervigilance);

    // MultiModalFusion::SensoryInput
    py::class_<MultiModalFusion::SensoryInput>(m, "SensoryInput")
        .def(py::init<>())
        .def_readwrite("visual", &MultiModalFusion::SensoryInput::visual)
        .def_readwrite("auditory", &MultiModalFusion::SensoryInput::auditory)
        .def_readwrite("vestibular", &MultiModalFusion::SensoryInput::vestibular)
        .def_readwrite("interoceptive", &MultiModalFusion::SensoryInput::interoceptive)
        .def_readwrite("timestamp", &MultiModalFusion::SensoryInput::timestamp)
        .def_readwrite("confidence", &MultiModalFusion::SensoryInput::confidence);

    // ImageToEmbedding
    py::class_<ImageToEmbedding>(m, "ImageToEmbedding")
        .def(py::init<const ImageToEmbedding::VisualConfig&>(), py::arg("config") = ImageToEmbedding::VisualConfig{})
        .def("process_image", &ImageToEmbedding::processImage, "Process visual input")
        .def("process_image_file", &ImageToEmbedding::processImageFile, "Process image from file")
        .def("process_simulated_scene", &ImageToEmbedding::processSimulatedScene, "Process simulated scene")
        .def("update_config", &ImageToEmbedding::updateConfig, "Update visual configuration")
        .def("add_ptsd_trigger_object", &ImageToEmbedding::addPTSDTriggerObject, "Add PTSD trigger object");

    // ImageToEmbedding::VisualConfig
    py::class_<ImageToEmbedding::VisualConfig>(m, "VisualConfig")
        .def(py::init<>())
        .def_readwrite("embedding_dimension", &ImageToEmbedding::VisualConfig::embedding_dimension)
        .def_readwrite("autism_detail_focus", &ImageToEmbedding::VisualConfig::autism_detail_focus)
        .def_readwrite("autism_face_processing_deficit", &ImageToEmbedding::VisualConfig::autism_face_processing_deficit)
        .def_readwrite("ptsd_threat_hypervigilance", &ImageToEmbedding::VisualConfig::ptsd_threat_hypervigilance)
        .def_readwrite("ptsd_startle_sensitivity", &ImageToEmbedding::VisualConfig::ptsd_startle_sensitivity);

    // AudioToEmbedding
    py::class_<AudioToEmbedding>(m, "AudioToEmbedding")
        .def(py::init<const AudioToEmbedding::AudioConfig&>(), py::arg("config") = AudioToEmbedding::AudioConfig{})
        .def("process_audio", &AudioToEmbedding::processAudio, "Process audio input")
        .def("process_audio_file", &AudioToEmbedding::processAudioFile, "Process audio from file")
        .def("process_simulated_audio", &AudioToEmbedding::processSimulatedAudio, "Process simulated audio")
        .def("update_config", &AudioToEmbedding::updateConfig, "Update audio configuration")
        .def("add_ptsd_trigger_sound", &AudioToEmbedding::addPTSDTriggerSound, "Add PTSD trigger sound")
        .def("add_combat_triggers", &AudioToEmbedding::addCombatTriggers, "Add combat triggers");

    // AudioToEmbedding::AudioConfig
    py::class_<AudioToEmbedding::AudioConfig>(m, "AudioConfig")
        .def(py::init<>())
        .def_readwrite("embedding_dimension", &AudioToEmbedding::AudioConfig::embedding_dimension)
        .def_readwrite("autism_auditory_hypersensitivity", &AudioToEmbedding::AudioConfig::autism_auditory_hypersensitivity)
        .def_readwrite("autism_volume_sensitivity", &AudioToEmbedding::AudioConfig::autism_volume_sensitivity)
        .def_readwrite("ptsd_hypervigilance", &AudioToEmbedding::AudioConfig::ptsd_hypervigilance)
        .def_readwrite("ptsd_combat_audio_sensitivity", &AudioToEmbedding::AudioConfig::ptsd_combat_audio_sensitivity);

    // VestibularSynth
    py::class_<VestibularSynth>(m, "VestibularSynth")
        .def(py::init<const VestibularSynth::VestibularConfig&>(), py::arg("config") = VestibularSynth::VestibularConfig{})
        .def("process_vestibular_input", &VestibularSynth::processVestibularInput, "Process vestibular input")
        .def("process_simulated_motion", &VestibularSynth::processSimulatedMotion, "Process simulated motion")
        .def("simulate_motion_pattern", &VestibularSynth::simulateMotionPattern, "Simulate motion pattern")
        .def("update_config", &VestibularSynth::updateConfig, "Update vestibular configuration")
        .def("calibrate_baseline", &VestibularSynth::calibrateBaseline, "Calibrate baseline");

    // InteroceptiveSim
    py::class_<InteroceptiveSim>(m, "InteroceptiveSim")
        .def(py::init<const InteroceptiveSim::InteroceptiveConfig&>(), py::arg("config") = InteroceptiveSim::InteroceptiveConfig{})
        .def("process_interoceptive_input", &InteroceptiveSim::processInteroceptiveInput, "Process interoceptive input")
        .def("process_simulated_body_state", &InteroceptiveSim::processSimulatedBodyState, "Process simulated body state")
        .def("simulate_physiological_state", &InteroceptiveSim::simulatePhysiologicalState, "Simulate physiological state")
        .def("simulate_stress_response", &InteroceptiveSim::simulateStressResponse, "Simulate stress response")
        .def("update_config", &InteroceptiveSim::updateConfig, "Update interoceptive configuration");

    // Utility functions
    m.def("create_autism_config", []() {
        NeuroSimulator::Config config;
        config.autism_mode = true;
        config.excitation_ratio = 1.4;
        config.inhibition_delay = 0.0;
        return config;
    }, "Create autism-specific configuration");

    m.def("create_ptsd_config", []() {
        NeuroSimulator::Config config;
        config.ptsd_overlay = true;
        config.inhibition_delay = 50.0;
        config.flashback_sensitivity = 0.5;
        return config;
    }, "Create PTSD-specific configuration");

    m.def("create_combined_config", []() {
        NeuroSimulator::Config config;
        config.autism_mode = true;
        config.ptsd_overlay = true;
        config.excitation_ratio = 1.4;
        config.inhibition_delay = 50.0;
        config.flashback_sensitivity = 0.6;
        return config;
    }, "Create combined autism + PTSD configuration");

    // Combat PTSD specific helper (for your background)
    m.def("create_combat_ptsd_config", []() {
        NeuroSimulator::Config config;
        config.ptsd_overlay = true;
        config.inhibition_delay = 60.0;  // Longer delay for combat PTSD
        config.flashback_sensitivity = 0.4;  // Higher sensitivity
        return config;
    }, "Create combat PTSD-specific configuration");

    m.def("add_fallujah_trauma_template", [](NeuroSimulator& sim) {
        // Create a trauma template based on Operation Phantom Fury context
        Eigen::VectorXd trauma_embedding = Eigen::VectorXd::Random(512);
        // In a real implementation, this would be based on actual combat scenarios
        sim.addTraumaMemory(trauma_embedding, 0.7);
    }, "Add Fallujah combat trauma template");
}

// Python loader helper
PYBIND11_MODULE(neurosim_loader, m) {
    m.doc() = "NeuroSim Engine Loader";
    
    m.def("load_neurosim", []() {
        return py::module::import("neurosim_py");
    }, "Load the NeuroSim Engine module");
    
    m.def("create_example_simulation", []() {
        auto config = NeuroSimulator::Config{};
        config.autism_mode = true;
        config.ptsd_overlay = true;
        config.log_level = "INFO";
        
        auto sim = NeuroSimulator(config);
        return sim;
    }, "Create an example simulation instance");
}

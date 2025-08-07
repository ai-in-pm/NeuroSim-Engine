#include "cerebellum.hpp"

// Stub implementation for cerebellum
// Owner: Darrell Mesa (darrell.mesa@pm-ss.org)

namespace neurosim {

Cerebellum::Cerebellum(const RegionConfig& region_config, 
                      const CerebellumConfig& cerebellum_config)
    : BrainRegion(region_config), cerebellum_config_(cerebellum_config) {
}

double Cerebellum::processInput(double input, double dt) {
    // Simple stub implementation
    current_activation_ = input * 0.3;
    return current_activation_;
}

} // namespace neurosim

//! The initial mob set: goal composition per archetype.
//!
//! A mob's "brain" is a `GoalSelector` filled with its goal stack. New mobs
//! register by composing existing goals (or adding new ones) — no central
//! match statement to grow unmaintainable.

use crate::ai::{self, GoalSelector};
use crate::world::{MobKind, MobState};

/// Build the AI brain for a mob kind.
pub fn brain(kind: MobKind) -> GoalSelector {
    let mut sel = GoalSelector::default();
    match kind {
        MobKind::Muncher => {
            // Passive: look around, wander, panic when hurt.
            sel.add(ai::panic(120.0));
            sel.add(ai::wander(MobKind::Muncher.speed()));
            sel.add(ai::look_around());
        }
        MobKind::Lurker => {
            // Hostile: melee when in reach, chase within sight, otherwise
            // wander/look. Panic is omitted (lurkers don't flee).
            sel.add(ai::melee(1.8, 20.0));
            sel.add(ai::chase(14.0, 20.0));
            sel.add(ai::wander(MobKind::Lurker.speed()));
            sel.add(ai::look_around());
        }
    }
    sel
}

/// Initial AI state per spawned mob (scratch cells shared by the goals:
/// [0] = wander/repath cooldown, [1] = panic timer, [2..3] = player pos,
/// [3] = melee attack signal). One array per mob keeps goals stateless.
pub fn new_scratch() -> [f32; 4] {
    [0.0; 4]
}

/// How much damage a mob's melee hit deals.
pub fn attack_damage(kind: MobKind) -> f32 {
    match kind {
        MobKind::Lurker => 3.0,
        MobKind::Muncher => 0.0,
    }
}

/// Hurt feedback hook: sets the panic flag on a passive mob.
pub fn on_hurt(state: &mut MobState, scratch: &mut [f32; 4], from_player: bool) {
    state.health -= 0.0; // the host applies the actual damage
    if from_player && !state.kind.is_hostile() {
        scratch[1] = 100.0; // panic for 5 s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brains_have_goals() {
        for kind in [MobKind::Muncher, MobKind::Lurker] {
            let brain = brain(kind);
            assert!(brain.running_count() == 0, "starts idle");
        }
    }
}

//! Hotbar + minimal inventory foundation.
//!
//! The hotbar is the visible 9-slot window into a simple slot list. Slots
//! hold block *names* resolved through the registry (ids are pack-specific;
//! names are stable). No crafting/drag-drop yet — this is the Phase-3
//! foundation: selection, wheel cycling, number keys, and placement counts.

/// Hotbar size (vanilla-compatible).
pub const HOTBAR_SIZE: usize = 9;

/// One inventory slot: a block name (or empty).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    pub block: Option<String>,
}

/// Linear inventory: hotbar (slots 0..9) plus a few reserve rows. Only the
/// hotbar is interactive in Phase 3.
pub struct Inventory {
    pub slots: Vec<Slot>,
    pub selected: usize,
}

impl Default for Inventory {
    fn default() -> Self {
        Self::new()
    }
}

impl Inventory {
    /// Fresh inventory with the sandbox palette in the hotbar.
    pub fn new() -> Inventory {
        let palette = [
            "stone",
            "dirt",
            "grass_block",
            "cobblestone",
            "oak_planks",
            "oak_log",
            "glass",
            "torch",
            "bricks",
        ];
        let mut slots: Vec<Slot> = (0..HOTBAR_SIZE)
            .map(|i| Slot {
                block: palette.get(i).map(|s| s.to_string()),
            })
            .collect();
        slots.resize(27, Slot { block: None }); // reserve rows
        Inventory { slots, selected: 0 }
    }

    /// The block name in the selected hotbar slot.
    pub fn selected_block(&self) -> Option<&str> {
        self.slots
            .get(self.selected)
            .and_then(|s| s.block.as_deref())
    }

    /// Select a slot (0..9). Out-of-range is ignored.
    pub fn select(&mut self, index: usize) {
        if index < HOTBAR_SIZE {
            self.selected = index;
        }
    }

    /// Cycle by wheel steps (positive = next).
    pub fn cycle(&mut self, steps: i32) {
        let n = HOTBAR_SIZE as i32;
        let next = (self.selected as i32 + steps).rem_euclid(n);
        self.selected = next as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotbar_starts_with_the_palette() {
        let inv = Inventory::new();
        assert_eq!(inv.slots.len(), 27);
        assert_eq!(inv.selected_block(), Some("stone"));
        assert_eq!(inv.slots[8].block.as_deref(), Some("bricks"));
        assert_eq!(inv.slots[9].block, None, "reserve row empty");
    }

    #[test]
    fn selection_by_number_and_wheel() {
        let mut inv = Inventory::new();
        inv.select(4);
        assert_eq!(inv.selected, 4);
        inv.select(9); // out of range: ignored
        assert_eq!(inv.selected, 4);
        inv.cycle(1);
        assert_eq!(inv.selected, 5);
        inv.cycle(-2);
        assert_eq!(inv.selected, 3);
        // Wrap-around both directions.
        inv.select(0);
        inv.cycle(-1);
        assert_eq!(inv.selected, HOTBAR_SIZE - 1);
        inv.cycle(1);
        assert_eq!(inv.selected, 0);
    }
}

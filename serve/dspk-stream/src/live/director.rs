//! One director, many viewers. Only the director configures, starts, stops and changes parameters.
//! Another client asks; with no director (or one whose kept role expired) it gets the role at
//! once, otherwise the director is asked and hands over, or not. A director who disconnects keeps
//! the role for [`DIRECTOR_GRACE`], so a dropped connection does not let anyone else take over a
//! running device; reconnecting in time gives it back.

use std::time::{Duration, Instant};

/// How long a disconnected director keeps its role.
pub const DIRECTOR_GRACE: Duration = Duration::from_secs(60);

/// What a request for control did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlOutcome {
    /// The caller is the director now (or already was).
    Granted,
    /// The director was asked to hand over.
    Asked { director: String },
}

#[derive(Debug, Clone, Default)]
pub struct Director {
    holder: Option<String>,
    /// The holder is disconnected; its role is kept until then.
    kept_until: Option<Instant>,
}

impl Director {
    /// The director (`None`: nobody, or a kept role that expired at `now`).
    pub fn holder(&mut self, now: Instant) -> Option<&str> {
        if self.kept_until.is_some_and(|until| now >= until) {
            self.holder = None;
            self.kept_until = None;
        }
        self.holder.as_deref()
    }

    pub fn is_director(&mut self, name: &str, now: Instant) -> bool {
        self.holder(now) == Some(name)
    }

    /// Until when a disconnected director's role is kept (`None`: connected, or no director).
    pub fn kept_until(&self) -> Option<Instant> {
        self.kept_until
    }

    /// `name` asks for control (it has the control permission: checked by the caller).
    pub fn request(&mut self, name: &str, now: Instant) -> ControlOutcome {
        match self.holder(now) {
            None => {
                self.holder = Some(name.to_string());
                self.kept_until = None;
                ControlOutcome::Granted
            }
            Some(h) if h == name => {
                self.kept_until = None;
                ControlOutcome::Granted
            }
            Some(h) => ControlOutcome::Asked { director: h.to_string() },
        }
    }

    /// The director gives the role to `to`. False when `from` is not the director.
    pub fn hand_over(&mut self, from: &str, to: &str, now: Instant) -> bool {
        if !self.is_director(from, now) {
            return false;
        }
        self.holder = Some(to.to_string());
        self.kept_until = None;
        true
    }

    /// The director gives the role up. False when `name` is not the director.
    pub fn release(&mut self, name: &str, now: Instant) -> bool {
        if !self.is_director(name, now) {
            return false;
        }
        self.holder = None;
        self.kept_until = None;
        true
    }

    /// `name` disconnected: a director keeps its role for [`DIRECTOR_GRACE`].
    pub fn disconnected(&mut self, name: &str, now: Instant) {
        if self.is_director(name, now) {
            self.kept_until = Some(now + DIRECTOR_GRACE);
        }
    }

    /// `name` connected: a director whose role was kept has it back.
    pub fn connected(&mut self, name: &str, now: Instant) {
        if self.is_director(name, now) {
            self.kept_until = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn granted_asked_handed_over() {
        let t = Instant::now();
        let mut d = Director::default();
        assert_eq!(d.request("a", t), ControlOutcome::Granted);
        assert_eq!(d.request("b", t), ControlOutcome::Asked { director: "a".into() });
        assert!(!d.hand_over("b", "b", t), "only the director hands over");
        assert!(d.hand_over("a", "b", t));
        assert!(d.is_director("b", t));
        assert!(d.release("b", t));
        assert_eq!(d.holder(t), None);
    }

    #[test]
    fn a_disconnected_director_keeps_the_role_for_a_while() {
        let t = Instant::now();
        let mut d = Director::default();
        d.request("a", t);
        d.disconnected("a", t);
        assert_eq!(d.request("b", t + Duration::from_secs(5)), ControlOutcome::Asked { director: "a".into() });
        // Back in time
        d.connected("a", t + Duration::from_secs(10));
        assert!(d.is_director("a", t + DIRECTOR_GRACE * 2));
        // Gone for longer than the grace period
        d.disconnected("a", t);
        assert_eq!(d.request("b", t + DIRECTOR_GRACE + Duration::from_secs(1)), ControlOutcome::Granted);
    }
}

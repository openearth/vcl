import pygame


class WipeTransition:
    """
    Left->right wipe:
      1) cover current with white (cover phase)
      2) swap images while fully covered
      3) reveal new image left->right (reveal phase)
    """

    def __init__(
        self, region: pygame.Rect, speed=1.2, color=(255, 255, 255), easing=None
    ):
        self.region = region
        self.speed = float(speed)  # wipes per second
        self.color = color
        self.easing = easing  # function t->t in [0,1], optional

        self.phase = "idle"
        self.progress = 0.0

        self.current = None
        self.next = None

    def set_current(self, surf: pygame.Surface):
        """Set the currently displayed image (size must match region.size)."""
        self.current = surf

    def start(self, next_surf: pygame.Surface):
        """Begin the transition to next_surf if idle."""
        if self.phase != "idle":
            return False
        self.next = next_surf
        self.phase = "cover"
        self.progress = 0.0
        return True

    def is_active(self):
        return self.phase != "idle"

    def _p(self):
        """Eased progress in [0,1]."""
        t = self.progress
        if self.easing:
            t = self.easing(t)
        return max(0.0, min(1.0, t))

    def update(self, dt: float):
        """Advance animation time."""
        if self.phase == "idle":
            return

        self.progress += self.speed * dt

        if self.progress >= 1.0:
            self.progress = 1.0

            if self.phase == "cover":
                # fully covered -> swap while hidden
                self.current = self.next
                self.next = None
                self.phase = "reveal"
                self.progress = 0.0
            elif self.phase == "reveal":
                self.phase = "idle"
                self.progress = 0.0

    def draw(self, screen: pygame.Surface):
        """Draw transition region. Call after drawing background."""
        if self.current is None:
            return

        r = self.region

        if self.phase == "idle":
            screen.blit(self.current, r.topleft)
            return

        p = self._p()
        w = int(r.width * p)

        if self.phase == "cover":
            # draw current
            screen.blit(self.current, r.topleft)
            # draw growing white cover
            if w > 0:
                cover_rect = pygame.Rect(r.left, r.top, w, r.height)
                pygame.draw.rect(screen, self.color, cover_rect)

        elif self.phase == "reveal":
            # draw full white cover, then reveal left part of new current
            pygame.draw.rect(screen, self.color, r)
            if w > 0:
                area = pygame.Rect(0, 0, w, r.height)
                screen.blit(self.current, r.topleft, area)


# Optional easing (smoothstep)
def ease_in_out(t):
    return t * t * (3 - 2 * t)


def main():
    pygame.init()
    screen = pygame.display.set_mode((900, 500))
    clock = pygame.time.Clock()

    region = pygame.Rect(80, 80, 420, 240)

    # Example surfaces (replace with loaded images)
    img_a = pygame.Surface(region.size)
    img_a.fill((30, 120, 240))
    pygame.draw.circle(img_a, (255, 200, 50), (120, 120), 70)

    img_b = pygame.Surface(region.size)
    img_b.fill((40, 180, 90))
    pygame.draw.rect(img_b, (240, 60, 80), (220, 60, 160, 140))

    # Create transition controller
    wipe = WipeTransition(region, speed=1.2, color=(255, 255, 255), easing=ease_in_out)
    wipe.set_current(img_a)

    # We'll toggle between A and B on space
    next_target = img_b

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                # Start transition without globals
                if wipe.start(next_target):
                    next_target = img_a if next_target is img_b else img_b

        # Update transition timing
        wipe.update(dt)

        # Draw scene
        screen.fill((18, 18, 18))
        pygame.draw.rect(screen, (80, 80, 80), region.inflate(8, 8), border_radius=10)

        wipe.draw(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

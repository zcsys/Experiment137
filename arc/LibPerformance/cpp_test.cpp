#include <SFML/Graphics.hpp>
#include <random>
#include <vector>
#include <algorithm>
#include <ranges>
#include <functional>

constexpr int WIDTH = 1920;
constexpr int HEIGHT = 1080;
constexpr int CIRCLE_COUNT = 100;
constexpr float CIRCLE_RADIUS = 10.f;
constexpr float MAX_SPEED = 5.f;

struct Entity {
    sf::CircleShape shape;
    sf::Vector2f velocity;
};

// Function to create a random float between min and max
auto randomFloat(float min, float max) -> float {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return static_cast<float>(dis(gen));
}

// Function to create a random entity
auto createRandomEntity() -> Entity {
    Entity entity;
    entity.shape.setRadius(CIRCLE_RADIUS);
    entity.shape.setPosition(randomFloat(0, WIDTH - 2 * CIRCLE_RADIUS),
                             randomFloat(0, HEIGHT - 2 * CIRCLE_RADIUS));
    entity.shape.setFillColor(sf::Color(
        static_cast<sf::Uint8>(randomFloat(0, 255)),
        static_cast<sf::Uint8>(randomFloat(0, 255)),
        static_cast<sf::Uint8>(randomFloat(0, 255))
    ));
    entity.velocity = sf::Vector2f(randomFloat(-MAX_SPEED, MAX_SPEED), randomFloat(-MAX_SPEED, MAX_SPEED));
    return entity;
}

// Function to update entity position
auto updateEntity(Entity& entity, float deltaTime) -> void {
    auto position = entity.shape.getPosition();
    position += entity.velocity * deltaTime;

    if (position.x < 0 || position.x > WIDTH - 2 * CIRCLE_RADIUS)
        entity.velocity.x = -entity.velocity.x;
    if (position.y < 0 || position.y > HEIGHT - 2 * CIRCLE_RADIUS)
        entity.velocity.y = -entity.velocity.y;

    entity.shape.setPosition(std::clamp(position.x, 0.f, WIDTH - 2 * CIRCLE_RADIUS),
                             std::clamp(position.y, 0.f, HEIGHT - 2 * CIRCLE_RADIUS));
}

auto main() -> int {
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Alife Simulation");
    window.setFramerateLimit(60);

    std::vector<Entity> entities(CIRCLE_COUNT);
    std::ranges::generate(entities, createRandomEntity);

    sf::Clock clock;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        float deltaTime = clock.restart().asSeconds();

        std::ranges::for_each(entities, [deltaTime](auto& entity) { updateEntity(entity, deltaTime); });

        window.clear(sf::Color::Black);
        std::ranges::for_each(entities, [&window](const auto& entity) { window.draw(entity.shape); });
        window.display();
    }

    return 0;
}

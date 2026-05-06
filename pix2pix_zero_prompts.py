from __future__ import annotations


def _default_animal_sentences(animal: str) -> list[str]:
    # ====== TODO 2 [必做-1]: 构造语义句子模板集合 ======
    # 目标:
    # 1) 给定 animal（如 cat / dog），生成多样化 prompt
    # 2) 覆盖场景、视角、姿态等变化，避免句子过于单一
    # 3) 返回去重后的句子列表

    templates: list[str] = [
        "a photo of a {animal}",
        "a close-up photo of a {animal}",
        "a portrait photo of a {animal}",
        "a high quality photo of a {animal}",
        "a detailed photo of a {animal}",
        "a photo of a {animal} sitting",
        "a photo of a {animal} standing",
        "a photo of a {animal} lying down",
        "a photo of a {animal} running",
        "a photo of a {animal} jumping",
        "a photo of a {animal} sleeping",
        "a photo of a {animal} playing",
        "a photo of a {animal} looking at the camera",
        "a photo of a {animal} looking to the side",
        "a photo of a {animal} indoors",
        "a photo of a {animal} outdoors",
        "a photo of a {animal} in a living room",
        "a photo of a {animal} in a kitchen",
        "a photo of a {animal} on a couch",
        "a photo of a {animal} on a bed",
        "a photo of a {animal} on a carpet",
        "a photo of a {animal} on a wooden floor",
        "a photo of a {animal} in a park",
        "a photo of a {animal} in a garden",
        "a photo of a {animal} on the street",
        "a photo of a {animal} on the grass",
        "a photo of a {animal} under a tree",
        "a photo of a {animal} near a window",
        "a photo of a {animal} in sunlight",
        "a photo of a {animal} in the shade",
        "a photo of a {animal} at night",
        "a photo of a {animal} in daylight",
        "a photo of a {animal} in the snow",
        "a photo of a {animal} on the beach",
        "a photo of a {animal} in the rain",
        "a photo of a {animal} with a blurred background",
        "a photo of a {animal} with bokeh background",
        "a face of a {animal}",
        "a headshot photo of a {animal}",
        "a photo of a {animal} with big eyes",
        "a photo of a {animal} with small ears",
        "a photo of a {animal} with short fur",
        "a photo of a {animal} with long fur",
        "a photo of a {animal} with a collar",
        "a photo of a {animal} wearing a collar",
        "a photo of a {animal} next to a person",
        "a photo of a {animal} next to a tree",
        "a photo of a {animal} next to a car",
        "a photo of a {animal} on a sidewalk",
        "a photo of a {animal} on a road",
        "a photo of a {animal} in a city",
        "a photo of a {animal} in the countryside",
        "a photo of a {animal} in a field",
        "a photo of a {animal} near water",
        "a photo of a {animal} by a lake",
        "a photo of a {animal} by a river",
        "a photo of a {animal} in a room",
        "a photo of a {animal} in a hallway",
        "a photo of a {animal} on stairs",
        "a photo of a {animal} under a table",
        "a photo of a {animal} near a chair",
        "a photo of a {animal} near a sofa",
        "a photo of a {animal} near a door",
        "a photo of a {animal} near a wall",
        "a photo of a {animal} on a blanket",
        "a photo of a {animal} on a rug",
        "a photo of a {animal} on a tile floor",
        "a photo of a {animal} on a concrete floor",
        "a photo of a {animal} on a balcony",
        "a photo of a {animal} in a backyard",
        "a photo of a {animal} in a front yard",
        "a photo of a {animal} close to the camera",
        "a photo of a {animal} far from the camera",
        "a photo of a {animal} centered in the frame",
        "a photo of a {animal} on the left side of the frame",
        "a photo of a {animal} on the right side of the frame",
        "a photo of a {animal} from above",
        "a photo of a {animal} from the side",
        "a photo of a {animal} from the front",
    ]

    prompts = [t.format(animal=animal) for t in templates]
    seen: set[str] = set()
    deduped: list[str] = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped
    # ====== END TODO 2 [必做-1] ======


def create_sentences() -> tuple[list[str], list[str]]:
    # ====== TODO 2 [必做-2]: 构造 source / target 句子集合 ======
    # 示例任务: cat -> dog
    # 可扩展为其它编辑方向（例如 dog -> wolf, sunny -> snowy 等）

    source_sentences = _default_animal_sentences("cat")
    target_sentences = _default_animal_sentences("dog")
    return source_sentences, target_sentences
    # ====== END TODO 2 [必做-2] ======

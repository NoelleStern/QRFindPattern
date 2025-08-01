import numpy as np
import argparse
import math
import cv2

from PIL import Image, ImageDraw
from typing import List
from enum import Enum


###
### The following code is based on this file: github.com/AlexandrGraschenkov/QRCodes/blob/master/qr/dot_detector.cpp
### My goal was to figure out how it works and touch up on it
###


_matchPattern: List[int] = [1, 1, 2.5, 1, 1] # Pattern scale

class LineType(Enum):
    Vertical = 1
    Horizontal = 2
    MajorDiagonal = 3
    MinorDiagonal = 4

# Contains a single line of an image and a potential finder pattern center
class ImageLine:
    array: np.ndarray = []
    centerIndex: int = 0

    def __init__(self, array, centerIndex) -> None:
        self.array = array
        self.centerIndex = centerIndex
    
    def __repr__(self) -> str:
        return f'ImageLine(array: {self.array}, centerIndex: {self.centerIndex})'

# A simple point class
class Point:
    x: int = 0
    y: int = 0

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    # Distant squared is faster
    def dist2(self, p):
        return (self.x-p.x)**2 + (self.y-p.y)**2
    
    def __repr__(self) -> str:
        return f'Point(x: {self.x}, y: {self.y})'

# Finder pattern position and size representation
class FinderPattern: 
    point: Point
    size: int = 0

    def __init__(self, point, size) -> None:
        self.point = point
        self.size = size
    
    def __repr__(self) -> str:
        return f'FinderPattern(point: {self.point}, size: {self.size})'


def main() -> None:
    parser: argparse.ArgumentParser = init_argparse()
    args: argparse.Namespace = parser.parse_args()
    filename: str = args.image[0]

    image: np.ndarray = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE)
    debugImage: np.ndarray = Image.open(filename).convert('RGBA')
    draw = ImageDraw.Draw(debugImage)

    result: List[FinderPattern] = findPatterns(image)
    for pattern in result:
        width: int = 2
        p: Point = pattern.point
        hs: int = pattern.size / 2 # Half size
        draw.rectangle([(p.x-hs, p.y-hs), (p.x+hs, p.y+hs)], outline='blue', width=width)
        draw.ellipse([(p.x-width, p.y-width), (p.x+width, p.y+width)], fill='blue')

    cv2.imshow('Image', image)
    cv2.imshow('Debug Image', np.array(debugImage))

    cv2.waitKey()
    cv2.destroyAllWindows()

# We're looking for the following pattern: b-w-2.5b-w-b
def checkPattern(aPattern: List[int], maxElemSize: int = 0, minElemSize: int = 0, variance: float = 0.4) -> bool:
    pattern: List[int] = [0, 0, 0, 0, 0]

    # Normalize the pattern using predefined scale
    # Ideally all of the values should get pretty close to each other
    for i in range(5):
        pattern[i] = aPattern[i] / _matchPattern[i]
    
    # Check element sizes if range is set
    if (maxElemSize > 0):
        for p in pattern:
            if (p > maxElemSize) or (p < minElemSize):
                return False
    
    # Count the total length in pixels
    total: int = 0
    for p in pattern:
        if (p == 0):
            return False
        
        total += pattern[i]
    
    avgSize: int = total / 5 # Average size
    maxVariance: float = avgSize * variance # Maximal allowed variance
    
    # Make sure variance is in check
    for p in pattern:
        if (abs(avgSize-p) > maxVariance):
            return False
    
    return True

# Shift the pattern so that the last three elements become first and we can continue searching
def shiftPattern(pattern: List[int]) -> List[int]:
    pattern[0] = pattern[2]
    pattern[1] = pattern[3]
    pattern[2] = pattern[4]
    pattern[3] = 0
    pattern[4] = 0

# Checks one line, but I'm not exactly 100% sure how it works
def checkOneLine(line: ImageLine, seqLength: int, variance: float = 2.5) -> bool:
    # Average element length
    elemSize: int = seqLength / (_matchPattern[0] +
                              _matchPattern[1] +
                              _matchPattern[2] +
                              _matchPattern[3] +
                              _matchPattern[4])

    sequence: List[int] = [0, 0, 0, 0, 0]
    idx: int = 2
    
    # Go through the row and find b-w-b-w-b sequences
    for y in range(line.centerIndex, len(line.array)):
        isWhite: bool = checkWhite(line.array[y])
        isWhiteIdx: bool = idx % 2

        if (isWhite != isWhiteIdx):
            idx += 1
            if (idx == 5): break

        sequence[idx] += 1
    
    idx = 2
    for y in range(line.centerIndex-1, 0, -1):
        isWhite: bool = checkWhite(line.array[y])
        isWhiteIdx: bool = idx % 2

        if (isWhite != isWhiteIdx):
            idx -= 1
            
            if (idx == -1):
                break

        sequence[idx] += 1

    # Now we perform a check with the size constraints
    return checkPattern(sequence, elemSize * variance, elemSize / variance)

# Checking the candidate vertically, horizontally and diagonally
def checkCandidate(image: np.ndarray, seqLength: int, center: Point) -> bool:
    hl: ImageLine = getLine(image, center, LineType.Horizontal)
    if not checkOneLine(hl, seqLength): return False

    vl: ImageLine = getLine(image, center, LineType.Vertical)
    if not checkOneLine(vl, seqLength): return False
    
    mjl: ImageLine = getLine(image, center, LineType.MajorDiagonal)
    if not checkOneLine(mjl, seqLength): return False
    
    mnl: ImageLine = getLine(image, center, LineType.MinorDiagonal)
    if not checkOneLine(mnl, seqLength): return False
    
    return True


# Making a one line matrix for convenience
def getLine(image: np.ndarray, center: Point, type: LineType) -> ImageLine:
    match (type):
        
        case LineType.Vertical:
            return ImageLine(image[:, center.x], center.y)

        case LineType.Horizontal:
            return ImageLine(image[center.y], center.x)
        
        # Positive offset is up, negative offset is down
        # Therefore we need to subtract y component from x
        case LineType.MajorDiagonal:
            centerIndex: int = min(center.x, center.y)
            offset: int = center.x-center.y
            return ImageLine(image.diagonal(offset), centerIndex)
        
        # The minor diagonal is a little bit stupid
        # To get it, we need to flip the matrix horizontally
        case LineType.MinorDiagonal:
            fx: int = image.shape[1]-center.x # Flipped x value
            centerIndex: int = min(fx, center.y)
            offset: int = fx-center.y
            return ImageLine(np.fliplr(image).diagonal(offset), centerIndex)

    return []

# Check if color is close enough to white
def checkWhite(color: int, threshold: int = 100):
    return color > threshold

# Fuses points together by distance
# https://stackoverflow.com/questions/19375675/simple-way-of-fusing-a-few-close-points
def fuse(patterns: List[FinderPattern], d: int = 100) -> List[FinderPattern]:
    res: List[FinderPattern] = []

    d2: int = d * d
    n: int = len(patterns)
    taken: List[bool] = [False] * n

    for i in range(n):
        if not taken[i]:
            count: int = 1
            pi: FinderPattern = patterns[i]
            pattern: FinderPattern = FinderPattern(Point(pi.point.x, pi.point.y), pi.size)
            taken[i] = True

            for j in range(i+1, n):
                pj: FinderPattern = patterns[j]

                if pi.point.dist2(pj.point) <= d2:
                    pattern.point.x += pj.point.x
                    pattern.point.y += pj.point.y
                    pattern.size += pj.size
                    
                    count += 1
                    taken[j] = True
            
            pattern.point.x /= count
            pattern.point.y /= count
            pattern.size /= count

            res.append(pattern)
    
    # Fix the floating numbers, kinda ugly that we have to do that
    result: List[FinderPattern] = []
    for p in res:
        result.append(FinderPattern(Point(int(p.point.x), int(p.point.y)), int(p.size)))

    return result

# Searches for finder patterns
# The image should be grayscale, step is a vertical step and we can safely check only every second line
def findPatterns(image: np.ndarray, step: int = 2) -> List[FinderPattern]:
    result: List[FinderPattern] = []
    
    # We're searching for a certain b-w-b-w-b pixel pattern
    for y in range(0, image.shape[0], step):

        #                      b, w, b, w, b
        sequence: List[int] = [0, 0, 0, 0, 0] # Stores length in pixels
        idx: int = 1 if checkWhite(image[y][0]) else 0
        
        for x in range(image.shape[1]):
            isWhite: bool = checkWhite(image[y][x]) # Actual color value
            idxIsWhite: bool = (idx % 2) # Color value according to index

            # Color has changed! This means we've just advanced!
            if (isWhite != idxIsWhite):
                idx += 1

                if (idx == 5):
                    # When the sequence is full, we check it
                    if (checkPattern(sequence)):
                        offset: int = sequence[4] + sequence[3] + math.ceil(sequence[2] / 2) + 1 # Half of pattern's length
                        seqLength: int = sequence[0] + sequence[1] + sequence[2] + sequence[3] + sequence[4] # The total length of sequence in pixels
                        center: Point = Point(x-offset, y)
                        #print(x, y, sequence)
                        
                        if checkCandidate(image, seqLength, center):
                            result.append( FinderPattern(center, seqLength) )
                    
                    # Now we can search further
                    # For that we only take the last b-w-b and continue from there
                    idx = 3 # Therefore idx is 3
                    shiftPattern(sequence)

            sequence[idx] += 1 # Increment pixel counter

    return fuse(result)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description = 'Finds edges of a QR code',
    )

    parser.add_argument('image', nargs='+', help='Image path')
    return parser


if __name__ == '__main__':
    main()
# clean-architecture-fastapi
## Introduction

Build testable, scalable and maintainable Python applications with Clean Architecture by leveraging FastAPI capabilities

Creating applications that are not only functional but also maintainable and scalable is paramount. Clean Architecture, conceptualized by Robert C. Martin (Uncle Bob), promotes a clear distinction between business logic and implementation details. This approach allows developers to focus on the essence of their applications without being tied to specific frameworks or external dependencies.

For this step-by-step tutorial, we’ll dive into the domain of auction management, and implement a piece of functionality that an auction management system might have. We’ll explore how TDD goes along with developing a Clean Architecture app. Additionally, we’ll try to integrate FastAPI, one of the most popular Python web frameworks, into the Clean Architecture paradigm.

## Get down to coding
The code used in this tutorial can be found in this github repository.

Let’s start with the central component of the architecture where the core business objects reside — the domain. We’ll define Item, Bid, and Auction entities.
```
# src/domain/enitites/item.py

from uuid import UUID, uuid4
from dataclasses import dataclass, field

@dataclass
class Item:
    name: str
    description: str
    id: UUID = field(default_factory=uuid4)
```

```
  # src/domain/enitites/bid.py

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from domain.value_objects.price import Price


@dataclass
class Bid:
    bidder_id: UUID
    price: Price
    auction_id: UUID
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
```

```
  # src/domain/enitites/auction.py

from dataclasses import dataclass, field
from datetime import date
from uuid import UUID, uuid4

from domain.enitites.bid import Bid
from domain.enitites.item import Item
from domain.value_objects.price import Price


@dataclass
class Auction:
    item: Item
    seller_id: UUID
    start_date: date
    end_date: date
    start_price: Price
    bids: list[Bid] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)

    @property
    def is_active(self) -> bool:
        return self.start_date <= date.today() <= self.end_date

    @property
    def minimal_bid_price(self) -> Price:
        return max(
            [bid.price for bid in self.bids] + [self.start_price], key=lambda x: x.value
        )
```

Within the Auction entity, we also use the Price object, which I’ve defined as a value object. How is a value object different from an entity? Value objects do not have unique identities. They are immutable over time: once created, their state can not be changed. In our example, if an auction price changes, we simply create a new one. Other examples of value objects include addresses, phone numbers, colors, or temperatures.

For now, we’ll consider accepting payments in only one currency.

Now that we’ve defined the basic data structures our app will operate on, we can begin writing our business logic — the use cases.

Here we may identify multiple scenarios, such as creating an item, creating an auction, submitting a bid for an existing auction, and so on. For the purpose of this tutorial, let’s focus on the latter.

Submitting a bid would involve the following logic:

+ Check if the auction the bid is submitted to exists.
+ Check if the auction is active.
+ Verify if the bid price is higher than the auction’s starting price or the highest bid.
+ Save the bid if all the conditions listed above are met.
At this point, let’s think of the adapters that we might need here, define the use case class signature, and proceed with TDD.

```
  # src/use_cases/submit_bid_use_case.py

from domain.enitites.bid import Bid
from ports.repositories.auction_repository import AuctionRepository


class SubmitBidUseCase:
    def __init__(self, auction_repository: AuctionRepository):
        self._auction_repository = auction_repository

    async def __call__(self, bid: Bid) -> None: ...
```

Within the use case, we’ll need to retrieve the auction by its identifier and save the bid if it meets our logic, regardless of where and how these entities are stored. To abstract the data access logic from the core business logic, we define AuctionRepositoryas an interface that will be implemented and tested separately.

```
  # src/ports/repositories/auction_repository.py

from abc import ABC, abstractmethod
from typing import Any

from domain.enitites.auction import Auction
from domain.enitites.bid import Bid


class AuctionRepository(ABC):
    @abstractmethod
    async def get(self, **filters: Any) -> Auction | None:
        pass

    @abstractmethod
    async def add_bid(self, bid: Bid) -> bool:
        pass

```

Before creating our first test, we also need to mock this layer. For this purpose, in-memory implementations are typically used: during testing, we store our entities in a simple Python list but interact with them using the same interface.
```
  # src/adapters/repositories/auction_repository/in_memory_repository.py

from typing import Any

from domain.enitites.auction import Auction
from domain.enitites.bid import Bid
from ports.repositories.auction_repository import AuctionRepository


class InMemoryAuctionRepository(AuctionRepository):
    auctions: list[Auction] = []

    async def get(self, **filters: Any) -> Auction | None:
        for auction in self.auctions:
            if (f := filters.get("id")) and f == auction.id:
                return auction
        return None

    async def add_bid(self, bid: Bid) -> bool:
        for auction in self.auctions:
            if auction.id == bid.auction_id:
                auction.bids.append(bid)
                return True
        return False
```
Now, let’s proceed with our first unit test. We’ll be using pytest and pytest-asyncio.

To inject the required dependencies (in our case, it’s only the auction repository), we’ll use pytest fixtures.

```
  # src/tests/unit/submit_bid_use_case_test.py

import uuid

import pytest

from adapters.repositories.auction_repository.in_memory_repository import (
    InMemoryAuctionRepository,
)
from ports.repositories.auction_repository import AuctionRepository
from tests.utils import create_bid
from use_cases.exceptions import AuctionNotFoundError
from use_cases.submit_bid_use_case import SubmitBidUseCase


@pytest.fixture
def auctions_repository() -> AuctionRepository:
    return InMemoryAuctionRepository()


@pytest.fixture
def submit_bid_use_case(auctions_repository: AuctionRepository) -> SubmitBidUseCase:
    return SubmitBidUseCase(auctions_repository)


async def test_auction_not_found(submit_bid_use_case: SubmitBidUseCase):
    bid = create_bid(auction_id=uuid.uuid4())
    with pytest.raises(AuctionNotFoundError):
        await submit_bid_use_case(bid)
```
In this first test, we expect that if a bid is submitted for an auction that does not exist (our in-memory storage is empty at this point), the corresponding exception will be raised.

If you attempt to run the test, it will obviously fail, as we have not yet implemented anything within the use case. Let’s do this.

```
  # src/use_cases/submit_bid_use_case.py

from domain.enitites.bid import Bid
from ports.repositories.auction_repository import AuctionRepository
from use_cases.exceptions import AuctionNotFoundError


class SubmitBidUseCase:
    def __init__(self, auction_repository: AuctionRepository):
        self._auction_repository = auction_repository

    async def __call__(self, bid: Bid) -> None:
        auction = await self._auction_repository.get(id=bid.auction_id)
        if not auction:
            raise AuctionNotFoundError(bid.auction_id)
```
Now, the test should run successfully.

By following this step-by-step development pattern (TDD), we build and test the rest of the logic described above for this particular use case.

```
  # src/tests/unit/submit_bid_use_case_test.py

...
async def test_auction_not_active(
    auctions_repository: InMemoryAuctionRepository,
    submit_bid_use_case: SubmitBidUseCase,
):
    auction = create_auction(end_date=date.today() - timedelta(days=1))
    auctions_repository.auctions.append(auction)
    bid = create_bid(auction_id=auction.id)
    with pytest.raises(AuctionNotActiveError):
        await submit_bid_use_case(bid)


async def test_bid_price_lower_than_start_price(
    auctions_repository: InMemoryAuctionRepository,
    submit_bid_use_case: SubmitBidUseCase,
):
    auction = create_auction(start_price_value=10)
    auctions_repository.auctions.append(auction)
    bid = create_bid(auction_id=auction.id, price_value=9)
    with pytest.raises(LowBidError):
        await submit_bid_use_case(bid)


async def test_bid_price_lower_than_highest_bid(
    auctions_repository: InMemoryAuctionRepository,
    submit_bid_use_case: SubmitBidUseCase,
):
    auction = create_auction(bids=[create_bid(price_value=20)])
    auctions_repository.auctions.append(auction)
    bid = create_bid(auction_id=auction.id, price_value=19)
    with pytest.raises(LowBidError):
        await submit_bid_use_case(bid)


async def test_bid_successfully_added(
    auctions_repository: InMemoryAuctionRepository,
    submit_bid_use_case: SubmitBidUseCase,
):
    auction = create_auction()
    auctions_repository.auctions.append(auction)
    bid = create_bid(auction_id=auction.id)
    await submit_bid_use_case(bid)
    result = await auctions_repository.get(id=auction.id)
    assert result is not None and bid in result.bids
```
Here is our final use case.
```
  # src/use_cases/submit_bid_use_case.py

...

class SubmitBidUseCase:
    def __init__(self, auction_repository: AuctionRepository):
        self._auction_repository = auction_repository

    async def __call__(self, bid: Bid) -> None:
        auction = await self._auction_repository.get(id=bid.auction_id)
        if not auction:
            raise AuctionNotFoundError(bid.auction_id)
        if not auction.is_active:
            raise AuctionNotActiveError(auction.id)
        if bid.price.value <= auction.minimal_bid_price.value:
            raise LowBidError(auction.minimal_bid_price)
        await self._auction_repository.add_bid(bid)
```
Let’s recap what we have so far. We’ve implemented the core business logic of the app and thoroughly tested it while setting aside the concerns of storing (SQL or NoSQL database, blob storage, etc.) and accepting (HTTP, RPC, message brokers, etc) data.

At this point the project structure looks like this:
```
  src
├── adapters
│   ├── exceptions.py
│   └── repositories
│       └── auction_repository
│           └── in_memory_repository.py
├── domain
│   ├── enitites
│   │   ├── auction.py
│   │   ├── bid.py
│   │   └── item.py
│   └── value_objects
│       └── price.py
├── ports
│   └── repositories
│       └── auction_repository.py
├── requirements.txt
├── tests
│   ├── conftest.py
│   ├── unit
│   │   └── submit_bid_use_case_test.py
│   └── utils.py
└── use_cases
    ├── exceptions.py
    └── submit_bid_use_case.py
```

We will discuss the pros and cons of this structure once we have completed the data access layer (our database storage adapter) and the drivers layer (the web framework).

It’s crucial to understand that, at this stage, our project does not have any third-party dependencies (any libraries installed with pip) except for pytest. And as our development progresses, we’ll never include any within our domain and use cases layers.

It’s time to develop a real implementation of our auction repository. To avoid managing the database schema and migration, I’ll opt for MongoDB and will store all the entities in one embedded document. As we are building an asynchronous app, we’ll need the motor library. Let’s install it and define the class signature for our adapter. Remember, it should implement the interface we already created.

```
  # src/adapters/repositories/auction_repository/mongodb_repository.py

from typing import Any

from domain.enitites.auction import Auction
from domain.enitites.bid import Bid
from ports.repositories.auction_repository import AuctionRepository
from motor.motor_asyncio import AsyncIOMotorClient


class MongoAuctionRepository(AuctionRepository):
    def __init__(self, client: AsyncIOMotorClient):
        self.collection = client.auctions.auction

    async def get(self, **filters: Any) -> Auction | None: ...

    async def add_bid(self, bid: Bid) -> bool: ...
```

Now lets run MongoDB docker container and get down to writing our first integration tests.
```docker run --rm -p 27017:27017 --name=mongo-auctions mongo:7.0.6```
```
  # src/tests/conftest.py

...
@pytest.fixture(scope="session")
def client():
    return AsyncIOMotorClient("mongodb://localhost:27017")
```

Of course we follow the same step-by-step TDD approach: one test per one peace of logic. Thus, we do not miss anything. For the purpose of testing, I created one more method within our repository to add auctions.

Here is what we eventually get.

```
  # src/tests/integration/repositories/mongodb_auction_repository_test.py

from uuid import uuid4

import pytest
from motor.motor_asyncio import AsyncIOMotorClient

from adapters.repositories.auction_repository.mongodb_repository import (
    MongoAuctionRepository,
)
from tests.utils import create_auction, create_bid


@pytest.fixture
async def auction_repository(client: AsyncIOMotorClient):
    repo = MongoAuctionRepository(client)
    yield repo
    await repo.collection.drop()


async def test_get_by_id_success(auction_repository: MongoAuctionRepository):
    auction = create_auction()
    await auction_repository.add(auction)
    assert await auction_repository.get(id=auction.id) == auction


async def test_get_by_id_not_found(auction_repository: MongoAuctionRepository):
    assert await auction_repository.get(id=uuid4()) is None


async def test_add_bid_success(auction_repository: MongoAuctionRepository):
    auction = create_auction()
    await auction_repository.add(auction)
    bid = create_bid(auction_id=auction.id)
    assert await auction_repository.add_bid(bid) is True


async def test_add_bid_auction_not_found(auction_repository: MongoAuctionRepository):
    bid = create_bid(auction_id=uuid4())
    assert await auction_repository.add_bid(bid) is False
```
```
  # src/adapters/repositories/auction_repository/mongodb_repository.py

...
class MongoAuctionRepository(AuctionRepository):
    def __init__(self, client: AsyncIOMotorClient):
        self.collection = client.auctions.auction 

    async def get(self, **filters: Any) -> Auction | None:
        filters = self.__get_filters(filters)
        try:
            document = await self.collection.find_one(filters)
            return self.__to_auction_entity(document) if document else None
        except Exception as e:
            raise DatabaseError(e)

    async def add_bid(self, bid: Bid) -> bool:
        try:
            r = await self.collection.update_one(
                {"_id": Binary.from_uuid(bid.auction_id)},
                {"$push": {"bids": self.__bid_to_doc(bid)}},
            )
            return bool(r.modified_count)
        except Exception as e:
            raise DatabaseError(e)

    async def add(self, auction: Auction) -> None:
        try:
            auction_doc = self.__auction_to_doc(auction)
            await self.collection.insert_one(auction_doc)
        except Exception as e:
            raise DatabaseError(e)

    @staticmethod
    def __get_filters(filters_args: dict[str, Any]) -> dict[str, Any]:
        filters = {}
        if f := filters_args.get("id"):
            filters["_id"] = Binary.from_uuid(f)
        return filters
...
```
As we’re not using any kind of ORM, we’ll need to define some methods for mapping our entities and MongoDB documents back and forth. I’ve left this behind the scene, as it is not the focus of this tutorial.

Now that we’ve completed our MongoDB implementation of the auction adapter, we can appreciate how easy it will be to switch to another technology: PostgreSQL, AWS S3 bucket, or anything else. We simply create another adapter, test it, and inject it into our use case without altering the core business logic.

Next, let’s consider an approach for accepting incoming data for a new bid. Implementing REST API built with FastAPI seems like a viable solution for our requirements. Powered by the Starlette, it handles every single request as an asynchronous task. This will allow us to benefit from the async app we already have. Let’s dive into it.
```
  # src/drivers/rest/main.py

from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, status, Depends
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from adapters.repositories.auction_repository.mongodb_repository import (
    MongoAuctionRepository,
)
from domain.enitites.bid import Bid
from domain.value_objects.price import CurrencyOption, Price
from drivers.rest.exception_handlers import exception_container
from ports.repositories.auction_repository import AuctionRepository
from use_cases.submit_bid_use_case import SubmitBidUseCase


app = FastAPI()


exception_container(app)


@lru_cache
def get_mongo_client() -> AsyncIOMotorClient:  # type: ignore
    return AsyncIOMotorClient("mongodb://localhost:27017")


def get_auction_repository(
    mongo_client: Annotated[AsyncIOMotorClient, Depends(get_mongo_client)],  # type: ignore
) -> AuctionRepository:
    return MongoAuctionRepository(mongo_client)


def get_submit_bid_use_case(
    auction_repository: Annotated[AuctionRepository, Depends(get_auction_repository)],
) -> SubmitBidUseCase:
    return SubmitBidUseCase(auction_repository)


class PriceInput(BaseModel):
    value: float
    currency: CurrencyOption

    def to_entity(self) -> Price:
        return Price(value=self.value, currency=self.currency)


class BidInput(BaseModel):
    bidder_id: UUID
    price: PriceInput

    def to_entity(self, auction_id: UUID) -> Bid:
        return Bid(
            auction_id=auction_id,
            bidder_id=self.bidder_id,
            price=self.price.to_entity(),
        )


@app.post("/auctions/{auction_id}/bids", status_code=status.HTTP_204_NO_CONTENT)
async def submit_bid(
    auction_id: UUID,
    data: BidInput,
    use_case: Annotated[SubmitBidUseCase, Depends(get_submit_bid_use_case)],
) -> None:
    await use_case(data.to_entity(auction_id))
```
For now, I’ve placed all the framework-related code in a single main.py file. However, as the service grows, maintaining it in this manner will become challenging. Therefore, you might want to check how I've structured it into multiple files in the project repository.

You might wonder why I opted for such a deeply nested FastAPI dependencies structure, with a separate function for every layer. The reason is flexibility. Let’s imagine we have an adapter that makes calls to a very expensive external API, and we want to call it only in production. With this approach, we can inject different implementations of the adapter depending on the environment:
```
  def get_super_expensive_api(
    settings: Annotated[Settings, Depends(get_settings)],
) -> SubmitBidUseCase:
    if settings.environment == "prod":
	return SuperExpensiveAPI()
    return FakeAPI()
```
Let’s get back to our example. We also should not forget to handle all the custom exceptions that may be raised within our app. Again, tests come first.
```
  Let’s get back to our example. We also should not forget to handle all the custom exceptions that may be raised within our app. Again, tests come first.

# src/drivers/rest/exception_handlers.py

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from use_cases.exceptions import (
    AuctionNotActiveError,
    AuctionNotFoundError,
    LowBidError,
)
from adapters.exceptions import ExternalError


def exception_container(app: FastAPI) -> None:
    @app.exception_handler(AuctionNotFoundError)
    async def auction_not_found_exception_handler(
        request: Request, exc: AuctionNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"message": str(exc)}
        )

    @app.exception_handler(AuctionNotActiveError)
    async def auction_not_active_exception_handler(
        request: Request, exc: AuctionNotActiveError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"message": str(exc)},
        )

    @app.exception_handler(LowBidError)
    async def low_bid_exception_handler(
        request: Request, exc: LowBidError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"message": str(exc)},
        )

    @app.exception_handler(ExternalError)
    async def external_exception_handler(
        request: Request, exc: ExternalError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "Something went wrong. Please try again"},
        )
```
```
  # src/drivers/rest/main.py

...
app = FastAPI()


exception_container(app)
...
```
Note that the framework dependencies (any FastAPI imports) should reside only within the drivers layer and nowhere else.

Now you can run the server and call our endpoint.
```
  uvicorn drivers.rest.main:app --reload

curl --location 'http://localhost:8000/auctions/6d90c671-a271-4075-9a42-73c8ddb3e65d/bids' \
--header 'Content-Type: application/json' \
--data '{
    "bidder_id": "3c6263c4-2729-4bb9-aa43-3154b51b1c23",
    "price": {
        "value": 90,
        "currency": "EUR"
    }
}'
```
You should get a 404 response indicating that this auction doesn’t exist yet. So go to the mongo shell and create it.
```
  docker exec -ti mongo-auctions mongosh

> use auctions

> db.auction.insertOne({'item': {'name': 'Test item', 'description': 'Test item description', '_id': UUID('ecdc079b-a409-4d9e-b0be-f6c170294195')}, 'seller_id': UUID('9b68b6dc-316b-4c2b-ade4-6cf8dafc8e01'), 'start_date': '2024-03-10', 'end_date': '2024-03-24', 'start_price': {'value': 10, 'currency': 'EUR'}, 'bids': [], '_id': UUID('6d90c671-a271-4075-9a42-73c8ddb3e65d')})
```
If you run the curl command again, you should receive a 204 response, and the auction bids will include the one we have just submitted.

Now, it would be beneficial to write some end-to-end tests. I usually use Postman for this purpose. However, let’s leave this outside the scope of this tutorial.

Here is the project structure we have so far, with FastAPI code divided into multiple modules and tests created for the endpoint:
```
  src
├── adapters
│   ├── exceptions.py
│   └── repositories
│       └── auction_repository
│           ├── in_memory_repository.py
│           └── mongodb_repository.py
├── domain
│   ├── enitites
│   │   ├── auction.py
│   │   ├── bid.py
│   │   └── item.py
│   └── value_objects
│       └── price.py
├── drivers
│   └── rest
│       ├── app.py
│       ├── dependencies.py
│       ├── exception_handlers.py
│       ├── main.py
│       └── routers
│           ├── auction.py
│           └── schema.py
├── ports
│   └── repositories
│       └── auction_repository.py
├── requirements.txt
├── tests
│   ├── conftest.py
│   ├── integration
│   │   ├── drivers
│   │   │   ├── auction_submit_bid_test.py
│   │   │   └── conftest.py
│   │   └── repositories
│   │       └── mongodb_auction_repository_test.py
│   ├── unit
│   │   └── submit_bid_use_case_test.py
│   └── utils.py
└── use_cases
    ├── exceptions.py
    └── submit_bid_use_case.py
```


## Clean Architecture and FastAPI integration
As our demo application is completed, I want to highlight some points related to the integration of Clean Architecture and FastAPI. If you already have experience with Clean Architecture, you should know that it usually involves controllers and presenters responsible for the validation and serialization of incoming and outgoing data. In out setup this functionality is delegated to FastAPI (Pydantic specifically). For resolving the dependencies, we use FastAPI’s built-in dependency injection mechanism. And you’ll probably agree that this is what we love FastAPI for.

In my opinion, it’s a reasonable trade-off. But we have to keep in mind that when switching to some other driver, we might run into problems. We’ll probably still be able to use Pydantic for data validation and serialization. But we’ll need to come up with some other solution for resolving dependencies.

## One more time, keep it clean!
I’ve encountered numerous examples of “clean architecture” on Python when use cases included framework dependencies (like HTTPException or even FastAPI Depends object).

Why is it important to keep it clean? Let me illustrate this with a real-life scenario. Recently, we run into a situation where the volume of data we needed to process surged. Having this and some other reasons in mind we decided to switch from HTTP to a message broker.

Having framework dependencies within our use case would require to fully rebuild the use case containing a lot of code and update all the tests. However, with the architecture I advocate in this tutorial, we were worried about nothing but the consumer.

## Conclusion
To draw the bottom line, let’s consider all the pros and cons that this project setup and Clean Architecture in general bring into place.

### Why is this cool?

+ We are not dependent on any third-party library. None of the frameworks impose their way of building the app;
+ Our app is decoupled; its parts are interchangeable without the need to fully rebuild it if we want to switch to another database or framework;
+ Because of this, the testability is awesome. It’s easier to add new logic and test it;
+ We have a nice and highly scalable project layout where every component has its place.
### What helped us to achieve this?

+ The concept of dependency injection: adapters (in our case AuctionRepository) are injected into the use cases;
+ The concept of inversion of control: instead of using a real implementation of a repository in the use case , we define an abstract class which outlines interfaces to be used within the use case.
### What are the downsides?

+ The above-mentioned concepts might be tricky to understand for the Clean Architecture newcomers; the project structure might seem too complicated at first glance;
+ With all this decoupling, we really need to write more code.
## Homework
Finishing our tutorial, let’s look at our only use case logic. How can it evolve in the future? At some point, we’ll probably want to process bids submitted in different currencies. This will certainly require calling some third-party API to get currency exchange rates. For making external asynchronous HTTP calls, we will need some other dependencies such as aiohttp or httpx libraries. Or what if, in the near future, a new and more effective library appears? As you could already understand, this piece of code is perfect for being implemented within our adapters layer, while comparing already converted bid prices would reside within the use case itself. Let’s keep this as homework for this tutorial.
